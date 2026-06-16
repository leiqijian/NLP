"""
演示
    transformer 的decoder解码器
组成:
    N个解码器层：LSDA
        解码器层1:
            子层1：带掩码的多头自注意力 + Add&Norm
            子层2：多头交叉注意力(Q来源于decoder,KV来源encoder) + Add&Norm
            子层3：前馈网络 + Add&Norm
        解码器层2
        ...
        解码器层6
    LayerNorm层归一化(工程上在编码器最后添加了LN,原论文中编码器最后没有LN)
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_encoder import *

# 1.定义模型类，实现transformer的decoder层/解码器层
class MyTransformerDecoderLayer(nn.Module):
    """
    transformer的解码器层
    输入:
        x: (batch_size, tgt_seq_len, d_model),由tgt_input得到的添加位置编码的词向量
        encoder_output: (batch_size, src_seq_len, d_model),由encoder得到的编码结果
        src_mask: (batch_size, src_seq_len),源序列的填充掩码
        tgt_mask: (tgt_q_seq_len, tgt_k_seq_len),目标序列的因果掩码
    输出:
        decoder_output: (batch_size, tgt_seq_len, d_model),解码器的输出结果
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        # 2.初始化网络层，3个子层连接
        # 使用nn.ModuleList来封装N个子层连接
        # 注册子模块参数，支持设备迁移/反向传播/模型保存
        # self.sublayers = nn.ModuleList(
        #     [MySublayerConnection(d_model=d_model, dropout=dropout) for _ in range(3)]
        # )
        # 初始化子层连接1
        self.sublayer1 = MySublayerConnection(d_model=d_model, dropout=dropout)
        self.sublayer2 = MySublayerConnection(d_model=d_model, dropout=dropout)
        self.sublayer3 = MySublayerConnection(d_model=d_model, dropout=dropout)
        # 初始化带掩码的多头注意力机制
        self.mask_attention = MyMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        # 初始化多头注意力机制
        self.multi_head_attention = MyMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        # 初始化前馈网络层
        self.feed_forward = MyFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    # 2.forward方法
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数：
            x,tgt_input的词向量,右移一位的目标序列的表示，(batch_size, tgt_seq_len, d_model)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, src_seq_len),源序列的填充掩码
            tgt_mask: (tgt_seq_len, tgt_seq_len),目标序列的因果掩码
        """
        # 1.经过子层1：带掩码的多头自注意力self.sublayers[0]
        # sublayer1 = self.sublayers[0]
        # todo: 带掩码的多头自注意力层需要 padding_mask 和 causal mask
        x = self.sublayers1(x, lambda x: self.mask_attention(query=x, key=x, value=x, tgt_mask=tgt_mask)[0])
        # 2.经过子层2：多头交叉注意力self.sublayers[1]
        x = self.sublayers2(x, lambda x: self.multi_head_attention(query=x, key=encoder_output, value=encoder_output, padding_mask=src_mask)[0])
        # 3.经过子层3：前馈网络self.sublayers[2]
        x = self.sublayers3(x, self.feed_forward)
        return x

# 2.定义函数，测试MyTransformerDecoderLayer
def test_MyTransformerDecoderLayer():
    # 1.定义参数
    d_model = 512
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 20
    num_heads = 8
    d_ff = 2048
    # 2.创建张量,BSD
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    print(f"输入张量x: {x.shape}")
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    print(f"编码器输出encoder_output: {encoder_output.shape}")
    # 3.创建MyTransformerDecoderLayer对象
    decoder_layer = MyTransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    # 4.创建掩码矩阵
    # 源序列的填充掩码，src_mask: (batch_size, src_seq_len)
    src_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool)
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, src_seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        src_mask[i, :lens[i]] = False
    print(f"填充掩码src_mask: {src_mask.shape}")
    # 因果掩码，tgt_mask: (tgt_seq_len, tgt_seq_len)，
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len),1).to(dtype=torch.bool)
    print(f"因果掩码tgt_mask: {tgt_mask.shape}")

    # 5.前向传播
    output = decoder_layer(
        x,
        encoder_output,
        src_mask,
        tgt_mask
    )
    print(f"输出张量output: {output.shape}")  # (batch_size,tgt_seq_len,d_model)

# 3.定义模型类，实现transformer的decoder/解码器
class MyTransformerDecoder(nn.Module):
    """
    transformer的解码器
    输入:
        x: (batch_size, tgt_seq_len, d_model),由tgt_input得到的添加位置编码的词向量
        encoder_output: (batch_size, src_seq_len, d_model),由encoder得到的编码结果
        src_mask: (batch_size, src_seq_len),源序列的填充掩码
        tgt_mask: (tgt_seq_len, tgt_seq_len),目标序列的因果掩码
    输出:
        x: (batch_size, tgt_seq_len, d_model),解码器的输出结果
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, num_layers=6):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.num_layers = num_layers
        # 2.初始化N个解码器层
        # 使用nn.ModuleList来封装N个解码器层
        # 注册子模块参数，支持设备迁移/反向传播/模型保存
        self.decoder_layers = nn.ModuleList(
            [MyTransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # 3.初始化LayerNorm层归一化
        self.layer_norm = MyLayerNorm(d_model)
    # 2.forward方法
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数：
            x,tgt_input的词向量,右移一位的目标序列的表示，(batch_size, tgt_seq_len, d_model)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, src_seq_len),源序列的填充掩码
            tgt_mask: (tgt_seq_len, tgt_seq_len),目标序列的因果掩码
        """
        # 1.经过N个解码器层
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        # 2.经过LayerNorm层归一化
        x = self.layer_norm(x)
        return x

# 4.定义函数，测试MyTransformerDecoder
def test_MyTransformerDecoder():
    # 1.定义参数
    d_model = 512
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 20
    num_heads = 8
    d_ff = 2048
    # 2.创建张量,BSD
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    print(f"输入张量x: {x.shape}")
    # print(f"输入张量x: {x}")
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    print(f"解码器输出encoder_output: {encoder_output.shape}")
    # 3.创建MyTransformerDecoder对象
    decoder = MyTransformerDecoder(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        num_layers=12
    )
    # 4.创建掩码矩阵
    # 源序列的填充掩码，src_mask: (batch_size, src_seq_len)
    src_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool)
    # print(src_mask)
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, src_seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        src_mask[i, :lens[i]] = False
    print(f"填充掩码src_mask: {src_mask.shape}")
    # 因果掩码，tgt_mask: (tgt_seq_len, tgt_seq_len)，
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len),1).to(dtype=torch.bool)
    print(f"因果掩码tgt_mask: {tgt_mask.shape}")

    # 5.前向传播
    output = decoder(
        x,
        encoder_output,
        src_mask,
        tgt_mask
    )
    print(f"输出张量output: {output.shape}")  # (batch_size,tgt_seq_len,d_model)
    # print(f"输出张量output: {output}")  # (batch_size,tgt_seq_len,d_model)

# 测试
if __name__ == '__main__':
    test_MyTransformerDecoderLayer()
    test_MyTransformerDecoder()
