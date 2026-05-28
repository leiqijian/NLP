"""
演示两类掩码张量mask(bool类型)：
    填充掩码padding_mask: (batch_size, seq_len)，True表示被屏蔽，False表示不被屏蔽
    因果掩码causal_mask: (tgt_seq_len, tgt_seq_len), 上三角矩阵，True表示被屏蔽，False表示不被屏蔽
涉及到的API:
    torch.triu(m,diagonal=1)：生成上三角矩阵,因果掩码causal_mask
    torch.masked_fill：按掩码替换元素值
    plt.matshow：可视化矩阵热力图
需要掌握:
    torch.triu(m,diagonal=1)：生成上三角矩阵
    torch.masked_fill：按掩码替换元素值

注意：
    掩码张量都是2d的，后续在模型内部才升维度成3d

"""
# 导包
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.定义函数，生成填充掩码张量padding_mask: (batch_size, seq_len)
def demo01():
    # 0.设置随机种子
    torch.manual_seed(4)
    # 1.定义参数
    batch_size = 6
    seq_len = 10
    # 2.初始化填充掩码张量为全1张量
    # padding_mask: (batch_size, seq_len)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # print(padding_mask)
    # 3.随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5,seq_len+1,(batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        padding_mask[i, :lens[i]] = False
    # print(padding_mask)
    # 4.可视化掩码矩阵
    plt.matshow(padding_mask)
    plt.title("填充掩码矩阵")
    plt.show()

# 2.定义函数，生成因果掩码张量causal_mask: (tgt_seq_len, tgt_seq_len)
def demo02():
    # 1.定义参数
    seq_len = 10
    # 2.创建因果掩码张量(seq_len,seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)
    # print(torch.ones(seq_len, seq_len))
    # print(causal_mask)
    # 3.可视化因果掩码矩阵
    plt.matshow(causal_mask)
    plt.title("因果掩码矩阵")
    plt.show()

# 3.定义函数，演示masked_fill实现将掩码张量的True替换为-1e9
def demo03():
    # 1.创建张量，模拟注意力分数scores = QK^T/sqrt(d_k), (batch_size, tgt_seq_len, src_seq_len)
    seq_len = 10
    batch_size = 2
    scores = torch.randn(batch_size, seq_len, seq_len)
    print(f"原始注意力分数scores:{scores.shape}")
    # 2.创建掩码张量，因果掩码：(seq_len, seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)
    # 3.使用masked_fill函数把掩码张量的True替换为-1e9
    scores = torch.masked_fill(scores, causal_mask==True, -1e9)
    print(f"替换后的注意力分数scores:{scores.shape}")
    # 4.可视化注意力分数矩阵热力图
    plt.matshow(scores[0])
    plt.title("注意力分数矩阵")
    plt.colorbar()
    plt.show()
    # 5.可视化注意力权重矩阵热力图
    attention_weights = torch.softmax(scores, dim=-1)
    plt.matshow(attention_weights[0])
    plt.title("注意力权重矩阵")
    plt.colorbar()
    plt.show()

def demo04():
    seq_len = 10
    batch_size = 2
    scores = torch.randn(batch_size, seq_len, seq_len)
    print(f"原始注意力分数scores:{scores.shape}")
    print(scores[0].shape) # scores[0] 取的是第0维（batch维）的第1个元素，可以看作是scores[0, :, :],即第一个批次的矩阵

# 5.定义函数，演示masked_fill实现将填充掩码张量的True替换为-1e9
def demo05():
    torch.manual_seed(4)
    # 1.创建张量，模拟注意力分数scores = QK^T/sqrt(d_k), (batch_size, seq_len, seq_len)
    seq_len = 10
    batch_size = 6
    scores = torch.randn(batch_size, seq_len, seq_len)
    print(f"原始注意力分数scores:{scores.shape}")
    # 2.创建填充掩码张量padding_mask: (batch_size, seq_len)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    lens = torch.randint(5, seq_len + 1, (batch_size,))
    for i in range(batch_size):
        padding_mask[i, :lens[i]] = False
    print(f"填充掩码padding_mask:{padding_mask.shape}")
    print(f"各样本有效长度: {lens.tolist()}")
    # 3.升维padding_mask为4D, (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(2)
    # 4.使用masked_fill函数把掩码张量的True替换为-1e9
    scores = torch.masked_fill(scores, padding_mask_4d, -1e9)
    print(f"替换后的注意力分数scores:{scores.shape}")
    # 5.可视化每个样本的注意力权重矩阵热力图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        attention_weights = torch.softmax(scores[i], dim=-1)
        im = ax.matshow(attention_weights.detach())
        ax.set_title(f"样本{i}(有效长度={lens[i].item()})")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("填充掩码后的注意力权重矩阵（各样本）")
    plt.tight_layout()
    plt.show()

# 测试
if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    # demo04()
    demo05()