"""
演示
    transformer输入部分的 词嵌入层 和 位置编码
输入部分组成：
    词嵌入层：把词转为词向量
    位置编码：添加序列位置信息到词向量中
需要掌握的:
    nn.Embedding
"""

# 导包
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 设置设备
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"当前设备：{device}")

# 1.定义模型类，实现词嵌入层 - 要求手敲
class MyEmbedding(nn.Module):
    """
    输入:
        x: 词索引, (batch_size, seq_len)
    输出:
        x: 词向量, (batch_size, seq_len, d_model)
    """
    # 1.定义__init__方法
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        # 1.初始化参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 2.初始化词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
    # 2.forward方法
    def forward(self, x):
        # 输入x: 词索引, (batch_size, seq_len)
        # 1.传入词嵌入层
        x = self.embedding(x)   # (batch_size, seq_len, d_model)
        # 2.进行缩放*d_model**0.5
        return x*self.d_model**0.5

# 2.测试词嵌入层
def test_MyEmbedding():
    # 1.定义参数
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 20
    # 2.创建MyEmbedding对象
    embedding = MyEmbedding(vocab_size, d_model)
    # 3.定义张量,模拟输入词索引序列,(batch_size,seq_len)
    x = torch.randint(0, vocab_size, (batch_size,seq_len))
    print(f"输入词索引序列x: \n{x.shape}")
    # 4.传入词嵌入层
    x = embedding(x)    # (batch_size, seq_len, d_model)
    print(f"输出词向量x: \n{x.shape}")

# 3.定义模型类,实现位置编码 - 要求手敲
class MyPositionalEncoding(nn.Module):
    """
    输入:
        x:词向量,(batch_size,seq_len,d_model)
    输出:
        x:添加位置编码后的词向量,(batch_size,seq_len,d_model)
    """
    # 1.定义__init__方法
    # d_model : 词向量长度
    # max_len ： 可输入句子的最大长度
    def __init__(self, d_model=512, dropout=0.1, max_len=1000):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            # 定义一个恒等层,实现什么都不做, y=x
            self.dropout = nn.Identity()
        self.max_len = max_len
        # 2.定义位置编码,(max_len, d_model)，初始化位置编码，设置为全0
        pe = torch.zeros(max_len, d_model)
        # 3.定义位置编码的索引序列pos,0~max_len-1
        # torch.arange(0, max_len)形状是1D的()
        pos = torch.arange(0, max_len).unsqueeze(1) # unsqueeze(1) 使得pos的形状为(max_len,1)， 后续才能跟div_term做矩阵乘法
        # 4.定义参数，记录10000^(-2i/d_model)
        # a^n = exp(log(a))^n = exp(log(a)*n)
        # 10000^(-2i/d_model) = exp(log(10000)*(-2i/d_model))
        div_term = torch.exp(np.log(10000.0)*(-torch.arange(0, d_model, 2) / d_model))
        # 5.计算位置编码pe, (max_len, d_model)
        pe[:, ::2] = torch.sin(pos * div_term)   # 偶数列的值
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数列的值
        # 6.把2D转为3D, (1,max_len,d_model),方便后面跟词嵌入层输出部分做张量加法，涉及到广播机制
        pe = pe.unsqueeze(0)
        # 7.将位置编码添加到缓冲区buffer: 不会训练，但必须跟着模型一起保存和移动的变量
        # 等价于self.pe = pe，并且实现了保存模型参数时一起保存，如果写self.pe = pe，则不会作为参数保存
        self.register_buffer("pe", pe)  # 将pe创建为一个self.pe的变量
    # 2.forward方法
    def forward(self, x):
        # 输入x: 词向量,(batch_size,seq_len,d_model)
        # 1.词向量(batch_size,seq_len,d_model) + 位置编码(batch_size,max_len,d_model)
        # self.pe[:, :x.size(1)] 把pe统一成x的维度，主要是第2个轴的长度保持一致才能相加等价于self.pe[:, :x.size(1), : ]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) # 防止过拟合

# 4.测试MyPositionalEncoding
def test_MyPositionalEncoding():
    # 1.定义参数
    vocab_size = 1000
    d_model = 512
    batch_size = 4
    seq_len = 20
    # 2.创建MyEmbedding对象
    embed = MyEmbedding(vocab_size, d_model)
    # 3.定义变量，模拟输入词索引序列，(batch_size,seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"x1: {x}")
    print(f"输入序列x: {x.shape}")
    # 4.输入词嵌入网络，得到词向量
    x = embed(x)
    print(f"x2: {x}")
    print(f"词向量x: {x.shape}")
    # 5.定义位置编码层
    pos_enc = MyPositionalEncoding(d_model)
    # 6.输入位置编码器层
    x = pos_enc(x)
    print(f"x3: {x}")
    print(f"添加位置编码后的词向量x: {x.shape}")

# 5.可视化位置编码pe
def show_pe():
    # 0.定义参数
    d_model = 16
    max_len = 1000
    seq_len = 100
    batch_size = 4
    # 1.创建全零张量，模拟输入词向量，(batch_size,seq_len,d_model)
    x = torch.zeros(batch_size, seq_len, d_model)
    # 2.创建位置编码层
    pos_enc = MyPositionalEncoding(
        d_model=d_model,
        dropout=0,
        max_len=max_len
    )
    # 3.输入位置编码层
    y = pos_enc(x)
    # 4.绘制位置编码曲线
    plt.figure(figsize=(12, 6))
    embed_range = np.arange(5,8)
    plt.plot(np.arange(seq_len), y[0,:,embed_range].detach())
    plt.legend(["d_model="+str(i) for i in embed_range])
    plt.show()

# 测试
if __name__ == "__main__":
    test_MyEmbedding()
    test_MyPositionalEncoding()
    show_pe()