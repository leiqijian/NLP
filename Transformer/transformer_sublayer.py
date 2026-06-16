"""
演示
    transformer编码器的 两个子层
    其实实现的是通用的encoder/decoder的子层
编码器子层1：
    Multi-Head Attention(多头注意力机制)
    Add & Norm(残差连接 + 层归一化)
编码器子层2：
    FeedForward(前馈网络层)
    Add & Norm(残差连接 + 层归一化)

需要掌握的:
    1.model(x, my_attention)
    2.lambda

"""
# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_encoder_element import *

# 1.定义模型类，实现子层连接结构
class MySublayerConnection(nn.Module):
    """
    实现transformer中的子层连接结构，子层对象(多头注意力机制/前馈网络层) + AddNorm
    """
    # 1.定义__init__方法
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        # 2.初始化LayerNorm层
        self.layer_norm = MyLayerNorm(normalized_shape=d_model) # layerNorm 是对一条样本的不同特征进行归一化
        # 3.dropout层
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    # 2.forward方法
    def forward(self, x, sublayer):
        # sublayer: 接收多头注意力神经网络 / 前馈网络层
        # 子层连接中组件的顺序SDAL: sublayer - dropout - Add - LayerNorm
        # 实际工业实现LSDA: LayerNorm - sublayer - dropout - Add
        return self.dropout(sublayer(self.layer_norm(x))) + x

# 2.定义函数，测试MySublayerConnection
def test_MySublayerConnection():
    # 1.定义参数
    d_model = 512
    batch_size = 4
    seq_len = 10
    num_heads = 8
    d_ff = 2048
    # 2.创建张量,BSD
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入张量x: {x.shape}")
    # 3.创建MySublayerConnection对象
    model = MySublayerConnection(d_model=d_model)
    # 4.前向传播，需要创建sublayer
    # 思路1: 函数对象传参
    # 多头注意力机制子层的前向传播，函数对象传参
    # def my_attention(x):
    #     model = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    #     return model(query = x, key = x, value = x)[0] 输入qkv拿到output
    # output = model(x, sublayer = my_attention)
    # 思路2: lambda函数传参
    # model是 MySublayerConnection 实例，即子层连接结构，负责包装 "LayerNorm → Sublayer → Dropout → Add(残差)" 这个流程
    # x 是原始输入
    # lambda x: MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)(query = x, key = x, value = x)[0]
    #  返回了一个MyMultiHeadAttention类对象
    output = model(x, sublayer = lambda x: MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)(query = x, key = x, value = x)[0])
    print(f"输出张量output: {output.shape}")    # (4, 10, 512)
    # 5.前馈网络层的前向传播
    # def my_feed_forward(x):
    #     return MyFeedForward(d_model=d_model, d_ff=d_ff)(x)
    output = model(output, lambda x: MyFeedForward(d_model=d_model, d_ff=d_ff)(x))
    print(f"输出张量output: {output.shape}")    # (4, 10, 512)

# 测试
if __name__ == "__main__":
    test_MySublayerConnection()