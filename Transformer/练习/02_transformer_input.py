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


class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model = 512):
        super(MyEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        x = self.embedding(x)
        return x*self.d_model**0.5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)