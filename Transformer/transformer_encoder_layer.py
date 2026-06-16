"""
演示
    transformer的encoder层，编码器层
组成:
    编码器层1：
        编码器子层1：LSDA
            Multi-Head Attention(多头注意力机制)
            Add & Norm(残差连接 + 层归一化)
        编码器子层2：
            FeedForward(前馈网络层)
            Add & Norm(残差连接 + 层归一化)
    编码器层2
    ...
    编码器层6
    LayerNorm(工程上在编码器最后添加了LN,原论文中编码器最后没有LN),因为工业上改成了LSDA,可能会对输出造成影响
    所以要在最后一层后额外加多一层LN

数据流:
    输入数据source:BS -> 词向量x: BSD -> encoder-output: BSD

"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_input import *
from transformer_encoder_element import *
from transformer_sublayer import *

# 1.定义模型类，实现transformer encoder层
class MyTransformerEncoderLayer(nn.Module):
    """
    transformer encoder层
    输入:
        x: 添加了位置编码的词向量,(batch_size,src_seq_len,d_model)
        padding_mask: 填充掩码,(batch_size,src_seq_len)
    输出:
        x: Encoder编码后的词向量,(batch_size,src_seq_len,d_model)
    """
    # 1.定义__init__方法
    def __init__(self,d_model=512,num_heads=8,d_ff=2048,dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model # qkv词向量维度
        self.num_heads = num_heads #头数
        self.d_ff = d_ff #隐藏层维度，用于前馈网络层定义隐藏层的神经元个数
        # 校验d_model 是否为n_heads的倍数
        assert self.d_model % self.num_heads == 0, f"d_model: {self.d_model}必须被num_heads: {self.num_heads}整除"
        # 初始化dropout层
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            # 定义一个恒等层,实现什么都不做, y=x
            self.dropout = nn.Identity()
        # 2.定义多头注意力层
        self.multi_head_attention = MyMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        # 3.定义前馈网络层
        self.feed_forward = MyFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        # 4.定义子层连接
        self.sublayer1 = MySublayerConnection(
            d_model=d_model,
            dropout=dropout
        )
        self.sublayer2 = MySublayerConnection(
            d_model=d_model,
            dropout=dropout
        )
    # 2.forward方法
    def forward(self, x, padding_mask=None):
        # 输入x: (batch_size,src_seq_len,d_model)
        # 1.经过子层1
        x = self.sublayer1(
            x, lambda x: self.multi_head_attention(
                x, x, x, padding_mask=padding_mask)[0]
        )
        # 2.经过子层2
        x = self.sublayer2(
            x, lambda x: self.feed_forward(x)
        )
        return x

# 2.定义函数，测试MyTransformerEncoderLayer
def test_MyTransformerEncoderLayer():
    # 1.定义参数
    d_model = 512
    batch_size = 4
    seq_len = 10
    num_heads = 8
    d_ff = 2048
    # 2.创建张量,BSD
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入张量x: {x.shape}")
    # 3.创建MyTransformerEncoderLayer对象
    encoder_layer = MyTransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    # 4.创建掩码矩阵，填充掩码
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # print(padding_mask)
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        padding_mask[i, :lens[i]] = False
    print(f"填充掩码padding_mask: {padding_mask}")
    # 5.前向传播
    output = encoder_layer(x,padding_mask=padding_mask)
    print(f"输出张量output: {output.shape}")    # (batch_size,seq_len,d_model)

# 测试
if __name__ == "__main__":
    test_MyTransformerEncoderLayer()