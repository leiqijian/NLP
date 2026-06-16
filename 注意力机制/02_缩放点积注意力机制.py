"""
演示
    缩放点积注意力机制的实现，这是后面要学的Transformer架构的最常用的注意力机制
核心公式：
    Attention(Q,K,V) = softmax(Q·K^T/sqrt(d_k))·V
计算步骤：
1. 计算注意力分数：QK^T（Query @ Key^T）
2. 缩放：除以 sqrt(d_k) 防止梯度消失
3. Softmax：转换为注意力权重（概率分布）
4. 加权求和：用权重对 Value 加权求和，@

任务描述:
    自定义三个张量: Q,K,V,模拟由source序列生成的K,V, 由target序列生成的Q
    Q: BSD(batch_size, tgt_seq_len, d_k), (2, 3, 32)
    K: BSD(batch_size, src_seq_len, d_k), (2, 4, 32)
    V: BSD(batch_size, src_seq_len, d_v), (2, 4, 64)
    Q·K^T: BSS(batch_size, tgt_seq_len, src_seq_len), (2, 3, 4)
    Attention(Q,K,V): BSD(batch_size, tgt_seq_len, d_v), (2, 3, 64)

参数定义:
    batch_size = 2
    tgt_seq_len = 3
    src_seq_len = 4
    d_k = 32
    d_v = 64

"""

# 导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1.定义模型类，实现缩放点积注意力机制
class MyScaledDotProductAttention(nn.Module):
    # 1.定义__init__方法
    def __init__(self, d_k, dropout=0.1):
        # 1.初始化父类成员
        super().__init__()
        # 2.初始化参数
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    # 2.定义forward方法
    def forward(self, Q, K, V):
        """
        根据传入的Q,K,V，进行缩放点积注意力计算，Attention(Q,K,V) = softmax(Q·K^T/sqrt(d_k))·V。
        输出注意力结果
        :param Q: query, BSD(batch_size, tgt_seq_len, d_k)
        :param K: key, BSD(batch_size, src_seq_len, d_k)
        :param V: value, BSD(batch_size, src_seq_len, d_v)
        :return: Attention(Q,K,V): BSD(batch_size, tgt_seq_len, d_v)
        """
        # 1. 计算注意力分数：QK^T（Query @ Key^T）
        scores = Q @ K.transpose(-1, -2) # BSS(batch_size, tgt_seq_len, src_seq_len)
        print(f"scores: {scores}, shape: {scores.shape}")

        # 2. 缩放：除以 sqrt(d_k) 防止梯度消失
        scores = scores / self.d_k ** 0.5

        # 3. Softmax：转换为注意力权重（概率分布）
        attention_weights = torch.softmax(scores, dim=-1)

        # 4. 加权求和：用权重对 Value 加权求和，@
        output = attention_weights @ V  # BSD(batch_size, tgt_seq_len, d_v)
        return output, attention_weights


# 主程序
if __name__ == '__main__':
    # 1.定义参数
    batch_size = 2
    tgt_seq_len = 3
    src_seq_len = 4
    d_k = 32
    d_v = 64

    # 2.创建Q,K,V
    # Q: BSD(batch_size, tgt_seq_len, d_k), (2, 3, 32)
    # K: BSD(batch_size, src_seq_len, d_k), (2, 4, 32)
    # V: BSD(batch_size, src_seq_len, d_v), (2, 4, 64)
    Q = torch.randn(batch_size, tgt_seq_len, d_k, requires_grad=True)
    K = torch.randn(batch_size, src_seq_len, d_k)
    V = torch.randn(batch_size, src_seq_len, d_v)
    print(f"Q: {Q}, shape: {Q.shape}")
    print(f"K: {K}, shape: {K.shape}")
    print(f"V: {V}, shape: {V.shape}")

    # 3.创建缩放点积注意力机制的模型对象
    model = MyScaledDotProductAttention(d_k)

    # 4.前向传播
    output, attention_weights = model(Q=Q, K=K, V=V)
    print(f"output: {output}, shape: {output.shape}")   # BSD(batch_size, tgt_seq_len, d_v)
    print(f"attention_weights: {attention_weights}, shape: {attention_weights.shape}")  # BSS(batch_size, tgt_seq_len, src_seq_len)

    # 5.可视化权重矩阵
    plt.matshow(attention_weights[1].detach())
    # 显示颜色条
    plt.colorbar()
    plt.show()

    ...
