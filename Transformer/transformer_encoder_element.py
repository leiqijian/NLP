"""
演示
    transformer 的 Encoder, 实现Encoder编码器各个部分的功能
组成：
    N层堆叠
        编码器层1：
            子层1：
                Multi-Head Attention 多头自注意力机制
                Add & Norm 残差连接 & 层归一化
            子层2
                Feed Forward 前馈网络
                Add & Norm 残差连接 & 层归一化
        编码器层2：
            子层2：
                Multi-Head Attention 多头自注意力机制
                Add & Norm 残差连接 & 层归一化
            子层2
                Feed Forward 前馈网络
                Add & Norm 残差连接 & 层归一化
        ...
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_input import MyEmbedding, MyPositionalEncoding
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1.定义Attention类，实现缩放点积注意力机制
class MyAttention(nn.Module):
    """
    Transformer的 缩放点积注意力机制，这里没有训练的参数
    输入:
        query/key/value: (batch_size, seq_len, d_model)
        padding_mask(可选): (batch_size, seq_len),填充掩码，True=被遮蔽/掩码，False=没有被遮蔽
        tgt_mask(可选,Decoder专用): (tgt_seq_len, tgt_seq_len),目标序列的填充掩码，防止看到未来/遮蔽未来token，True=被遮蔽/掩码，False=没有被遮蔽
    返回:
        output: (batch_size, seq_len, d_model)
        attention_weights: (batch_size, seq_len, seq_len)即(batch_size, q_seq_len, k_seq_len)
    注意：
        关于掩码张量:模型外统一为2D张量，模型内部再升级成4D，以适配多头注意力机制的计算
        多头注意力机制的注意力分数形状为BNSS(batch_size,num_heads, q_seq_len, k_seq_len)
            解释：
                Q、K 在多头拆分后是 BNSH (batch_size, num_heads, seq_len, d_head)
                scores = Q @ K.transpose(-2, -1) → BNSH @ B NHS(T) = B N S S，即 (batch_size, num_heads, q_len, k_len)
                在编码器自注意力中 q_len == k_len，所以实际是 (batch_size, num_heads, seq_len, seq_len)

    """
    # 1.定义__init__方法
    # 缩放点积注意力机制没有任何可训练的参数
    def __init__(self, dropout=0.1):
        super().__init__()
        # 1.初始化dropout层
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            # 定义一个恒等层,实现什么都不做, y=x
            self.dropout = nn.Identity()
    # 2.forward方法
    def forward(self, query, key, value, padding_mask=None, tgt_mask=None):
        # query/key/value: (batch_size, seq_len, d_model),(batch_size,num_heads,seq_len,d_head)
        # padding_mask: (batch_size, k_seq_len)
        # tgt_mask: (q_len, k_seq_len)
        # 1.获取参数batch_size, seq_len, d_model
        batch_size = query.shape[0]
        d_model = query.shape[-1]
        q_len = query.shape[-2]
        k_len = key.shape[-2]
        # 2.计算缩放点积注意力，打分-处理掩码-归一化-加权求和
        scores = query @ key.transpose(-2, -1) / d_model**0.5 # (batch_size, q_len, k_len)
        # scores在多头注意力机制中是BNSS  (batch_size,num_heads, q_len, k_len)
        # 后续掩码升维度都是为了和sorce维度对齐
        # 3.处理掩码，填充掩码，因果掩码
        # 3.1 处理填充掩码, padding_mask(可选): (batch_size, k_len)
        if padding_mask is not None:
            # 1.维度校验
            if padding_mask.dim() == 2:
                # 2.校验形状(batch_size, k_len)
                # padding_mask是 被关注的对象（key/value）的属性，掩盖key中那些地方不需要被关注，所以是k_len
                assert padding_mask.shape == (batch_size, k_len), f"填充掩码的形状{padding_mask.shape}错误, 需为{(batch_size, k_len)}"
                # 3.扩展填充掩码为4D,(batch_size, k_len)-> (batch_size, 1, 1, k_len)
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            elif padding_mask.dim() == 3:
                # 4.扩展填充掩码为4D,(batch_size,*, k_len)-> (batch_size, 1, *, k_len)
                padding_mask = padding_mask.unsqueeze(1)
            scores = torch.masked_fill(scores, padding_mask, -1e9)
        # 3.2 处理因果掩码，tgt_mask(可选,Decoder专用): (q_len, k_len)
        if tgt_mask is not None:
            assert tgt_mask.shape == (q_len, k_len), f"因果掩码的形状{tgt_mask.shape}错误, 需为{(q_len, k_len)}"
            # 扩展因果掩码为4D,(q_len, k_len)-> (1, 1, q_len, k_len)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            scores = torch.masked_fill(scores, tgt_mask, -1e9)
        # 4.归一化
        # 对scores的最后一个轴做归一化
        # 原因:
        #   1. 把原始分数变成概率分布，每个 query 位置对所有 key 位置的权重之和为 1,突出权重
        #   2. scores 形状是 (batch_size, num_heads, q_len, k_len)，想要突出每个 query 位置（q_len 维）需要独立地决定它对所有 key 位置（k_len 维）的关注程度
        attention_weights = F.softmax(scores, dim=-1)
        # 5.加权求和
        # 返回注意力输出output和注意力权重attention_weights
        output = attention_weights @ value
        return output, attention_weights

# 2.测试MyAttention
def test_MyAttention():
    # 1.定义参数
    vocab_size = 1000
    d_model = 512
    batch_size = 4
    seq_len = 10
    # 2.创建MyEmbedding对象
    embed = MyEmbedding(vocab_size, d_model)
    # 3.定义变量，模拟输入词索引序列，(batch_size,seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入序列x: {x.shape}")
    # 4.输入词嵌入网络，得到词向量
    x = embed(x)
    print(f"词向量x: {x.shape}")
    # 5.定义位置编码层
    pos_enc = MyPositionalEncoding(d_model)
    # 6.输入位置编码器层
    x = pos_enc(x)
    print(f"添加位置编码后的词向量x: {x.shape}")
    # 7.创建MyAttention对象
    attention = MyAttention(dropout=0)
    # 8.前向传播，没有掩码的前向传播
    query, key, value = x, x, x
    output, attention_weights = attention(query, key, value)
    # output: (batch_size, seq_len, d_model)
    # attention_weights: (batch_size, seq_len, seq_len)
    print(f"输出output: {output.shape}")
    print(f"注意力权重attention_weights: {attention_weights.shape}")
    # print(f"注意力权重attention_weights: {attention_weights}")
    # 绘制注意力权重热力图
    plt.matshow(attention_weights[0].detach())
    plt.title("没有掩码的注意力权重")
    plt.show()
    # 9.添加因果掩码的前向传播
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)
    output, attention_weights = attention(
        query, key, value,
        tgt_mask=tgt_mask
    )
    # output: (batch_size, seq_len, d_model)
    # attention_weights: (batch_size, seq_len, seq_len)
    print(f"输出output: {output.shape}")
    print(f"注意力权重attention_weights: {attention_weights.shape}")
    print(f"注意力权重attention_weights[0,0]: {attention_weights[0,0].shape}")
    # print(f"注意力权重attention_weights: {attention_weights}")
    # 绘制注意力权重热力图
    plt.matshow(attention_weights[0,0].detach())
    plt.title("有掩码的注意力权重")
    plt.show()
    print("-" * 40)

# 3.定义多头注意力机制类
class MyMultiHeadAttention(nn.Module):
    """
    参数：
        d_model: QKV的特征维度，通常为512,1024,2048
        num_heads: 多头注意力的头数，通常为8,16, 要满足整除关系，d_model % num_heads == 0
        dropout: 可选，默认0.1
    输入:
        query/key/value: (batch_size, seq_len, d_model)
        padding_mask(可选): (batch_size, seq_len),填充掩码，True=被遮蔽/掩码，False=没有被遮蔽
        tgt_mask(可选,Decoder专用): (tgt_seq_len, tgt_seq_len),目标序列的填充掩码，防止看到未来/遮蔽未来token，True=被遮蔽/掩码，False=没有被遮蔽
    返回:
        output: (batch_size, seq_len, d_model)
        attention_weights: (batch_size, seq_len, seq_len)
    """
    # 1.定义__init_方法
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.num_heads = num_heads
        # 2.校验整数关系：d_model % num_heads == 0
        assert d_model % num_heads == 0, f"d_model={d_model}不能被num_heads={num_heads}整除!"
        self.d_head = d_model // num_heads
        # 3.定义线性层，4个线性层，query/key/value/output
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        # 4.定义缩放点积注意力机制
        self.attention = MyAttention(dropout=dropout)
        self.attention_weights = None   # 用于可视化注意力权重矩阵
    # 2.forward方法
    def forward(self, query, key, value, padding_mask=None, tgt_mask=None):
        # query/key/value: (batch_size, seq_len, d_model)
        # 0.获取batch_size
        batch_size = query.shape[0]
        # 1.经过线性层得到新的QKV
        query = self.linear_q(query)    # (batch_size, seq_len, d_model)
        key = self.linear_k(key)    # (batch_size, seq_len, d_model)
        value = self.linear_v(value)    # (batch_size, seq_len, d_model)
        # 2.拆分QKV为多个头
        # BSD(batch_size,seq_len,d_model) -> BSNH -> BNSH(batch_size,num_heads,seq_len,h_head)
        query = query.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2) # BNSH
        key = key.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2) # BNSH
        value = value.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2) # BNSH
        # 3.计算每个头的注意力
        output, self.attention_weights = self.attention(
            query, key, value,
            padding_mask=padding_mask,
            tgt_mask=tgt_mask
        )
        # output: BNSH(batch_size,num_heads,seq_len,d_head)
        # attention_weights: BNSS(batch_size,num_heads,seq_len,seq_len)
        # 4.拼接所有头输出
        # BNSH -> BSNH -> BSD(batch_size,seq_len,d_model)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)   # BSD(batch_size,seq_len,d_model)
        # 5.经过线性层
        output = self.linear_out(output)
        return output, self.attention_weights

# 4.测试MyMultiHeadAttention
def test_MyMultiHeadAttention():
    # 1.定义参数
    vocab_size = 1000
    d_model = 512
    batch_size = 4
    seq_len = 10
    # 2.创建MyEmbedding对象
    embed = MyEmbedding(vocab_size, d_model)
    # 3.定义变量，模拟输入词索引序列，(batch_size,seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入序列x: {x.shape}")
    # 4.输入词嵌入网络，得到词向量
    x = embed(x)
    print(f"词向量x: {x.shape}")
    # 5.定义位置编码层
    pos_enc = MyPositionalEncoding(d_model)
    # 6.输入位置编码器层
    x = pos_enc(x)
    print(f"添加位置编码后的词向量x: {x.shape}")
    # 7.创建MyMultiHeadAttention对象
    attention = MyMultiHeadAttention(dropout=0)
    # 8.前向传播，没有掩码的前向传播
    query, key, value = x, x, x
    output, attention_weights = attention(query, key, value)
    # output: (batch_size, seq_len, d_model)
    # attention_weights: (batch_size, seq_len, seq_len)
    print(f"输出output: {output.shape}")
    print(f"注意力权重attention_weights: {attention_weights.shape}")
    # print(f"注意力权重attention_weights: {attention_weights}")
    # 绘制注意力权重热力图
    plt.matshow(attention_weights[0,0].detach())
    plt.title("没有掩码的多头注意力权重")
    plt.show()
    # 9.添加因果掩码的前向传播
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)
    output, attention_weights = attention(
        query, key, value,
        tgt_mask=tgt_mask
    )
    # output: (batch_size, seq_len, d_model)
    # attention_weights: (batch_size, seq_len, seq_len)
    print(f"输出output: {output.shape}")
    print(f"注意力权重attention_weights: {attention_weights.shape}")
    print(f"注意力权重attention_weights[0,0]: {attention_weights[0,0].shape}")
    # print(f"注意力权重attention_weights: {attention_weights}")
    # 绘制注意力权重热力图
    plt.matshow(attention_weights[0,0].detach())
    plt.title("有掩码的多头注意力权重")
    plt.show()
    print("-" * 40)

# 5.定义前馈神经网络类
class MyFeedForward(nn.Module):
    """
    参数：
        d_model:特征维度，需要与MultiHeadAttention中保持一致
        d_ff: 中间层特征维度,自定义
        dropout: 可选，默认为0.1
    输入：
        x: (batch_size, seq_len, d_model)
    返回：
        output: (batch_size, seq_len, d_model)
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.d_ff = d_ff
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            # 定义一个恒等层,实现什么都不做, y=x
            self.dropout = nn.Identity()
        # 2.初始化线性层
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    # 2.forward方法
    def forward(self, x):
        # 输入x: (batch_size, seq_len, d_model), 注意力机制的输出
        # 1.经过线性层1+relu
        x = F.relu(self.linear1(x))
        # 2.经过dropout
        x = self.dropout(x)
        # 3.经过线性层2
        x = self.linear2(x)
        return x

# 6.测试前馈神经网络层MyFeedForward
def test_MyFeedForward():
    # 1.初始化参数
    batch_size = 4
    seq_len = 10
    d_model = 512
    d_ff = 2048
    # 2.创建输入数据，BSD
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入数据x: {x.shape}")
    # 3.创建前馈神经网络层
    model = MyFeedForward(d_model, d_ff)
    # 4.前向传播
    output = model(x)
    print(f"输出output: {output.shape}")
    print("-" * 40)

# 7.定义LayerNorm类
class MyLayerNorm(nn.Module):
    # 1.定义__init__方法
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # 1.初始化参数
        self.normalized_shape = normalized_shape
        self.eps = eps
        # 2.初始化可学习参数
        # 全1张量
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        # 全0张量
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    # 2.forward方法
    def forward(self, x):
        # 输入x: (batch_size, seq_len, d_model)
        # 1.计算均值和标准差
        mean = torch.mean(x, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        std = torch.std(x, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        # 2.进行标准化
        x = (x-mean)/(std**2+self.eps)**0.5
        # 3.进行缩放和平移
        x = self.gamma * x + self.beta
        return x

# 8.测试MyLayerNorm
def test_MyLayerNorm():
    # 1.定义参数
    batch_size = 4
    seq_len = 10
    d_model = 512
    # 2.创建数据，(batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)*100
    print(f"输入数据x: {x.shape}")
    print(f"输入数据x[0,0]: {x[0,0]}")
    # 3.创建MyLayerNorm对象
    model = MyLayerNorm(d_model)
    # 4.前向传播
    output = model(x)
    print(f"输出output: {output.shape}")
    print(f"输出output[0,0]: {output[0,0]}")
    print("-" * 40)

# 测试
if __name__ == '__main__':
    # test_MyAttention()
    # test_MyMultiHeadAttention()
    # test_MyFeedForward()
    test_MyLayerNorm()
