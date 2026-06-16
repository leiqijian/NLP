"""
演示
    transformer的编码器encoder

组成:
    N个编码器层堆叠
        编码器层1
        ...
        编码器层6
    LayerNorm(工程上在编码器最后添加了LN,原论文中编码器最后没有LN)

"""
# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_encoder_layer import *

# 1.定义模型类，实现完整的transformer encoder
class MyTransformerEncoder(nn.Module):
    """
    实现transformer的encoder编码器，就是N个编码器层堆叠 + LayerNorm
    输入:
        x: (batch_size, src_seq_len, d_model),添加位置编码的词向量
        padding_mask: (batch_size, src_seq_len),源序列的填充掩码
    输出:
        encoder_output: (batch_size, src_seq_len, d_model),编码器的输出
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model #QKV的特征维度，类比RNN的hidden_size
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers #编码器层数
        # if dropout is not None:
        #     self.dropout = nn.Dropout(dropout)
        # else:
        #     self.dropout = nn.Identity()    # 创建一个恒等层，就是y=x
        # 2.创建网络层
        # 使用nn.ModuleList来封装N个编码器层
        # 注册子模块参数，支持设备迁移/反向传播/模型保存
        self.encoder_layers = nn.ModuleList(
            [MyTransformerEncoderLayer(d_model = d_model, num_heads = num_heads, d_ff = d_ff, dropout = dropout) for _ in range(num_layers)]
        )# .to(device)
        # 3.创建层归一化
        self.layer_norm = MyLayerNorm(d_model)
    # 2.forward方法
    def forward(self, x, padding_mask=None):
        # x: (batch_size, src_seq_len, d_model), 添加位置编码的词向量
        # padding_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # 1.经过N个编码器层
        # 第1种写法
        # x = self.encoder_layers[0](x, padding_mask)
        # x = self.encoder_layers[1](x, padding_mask)
        # x = self.encoder_layers[2](x, padding_mask)
        # ...
        # 第2种写法
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask = padding_mask) #左边的x会被右边返回的x替代，进行下一轮的循环计算
        # 第3种写法
        # for i in range(self.num_layers):
        #     x = self.encoder_layers[i](x, padding_mask = padding_mask)
        # 2.经过层归一化
        x = self.layer_norm(x)
        return x

# 2.定义函数，测试MyTransformerEncoder
def test_MyTransformerEncoder():
    # 1.定义参数
    batch_size = 4
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    num_layers = 6
    # 2.创建MyTransformerEncoder对象
    model = MyTransformerEncoder(
        d_model, num_heads, d_ff, num_layers, dropout
    )
    # 3.创建张量，模拟输入数据，BSD
    x = torch.randn(batch_size, seq_len, d_model)*1000
    print(f"输入张量x: {x.shape}")
    # print(f"输入张量x: {x[0,0]}")
    # 4.创建填充掩码padding_mask
    # padding_mask: (batch_size, seq_len)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # print(padding_mask)
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        padding_mask[i, :lens[i]] = False
    print(f"填充掩码padding_mask: {padding_mask.shape}")
    # 5.输入编码器，前向传播
    encoder_output = model(x, padding_mask)
    print(f"编码器的输出encoder_output: {encoder_output.shape}")
    # print(f"编码器的输出encoder_output: {encoder_output[0,0]}")

# 测试
if __name__ == '__main__':
    test_MyTransformerEncoder()


'''
d_model类比RNN神经网络的构建rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1, batch_first=True)是input_size还是hidden_size

结论：d_model 更接近 RNN 的 hidden_size
     
  关键区别
     
  在 RNN 中，input_size 和 hidden_size 可以是不同的：
     
  rnn = nn.RNN(input_size=128, hidden_size=256)
  # 输入: (batch, seq, 128)
  # 输出: (batch, seq, 256)   ← 维度变了！

  RNN 内部做了维度映射：128 → 256。

  在 Transformer 中，d_model 贯穿始终，保持恒定：

  # transformer_encoder_element.py 第190-193行                                                                                                                                                         
  self.linear_q = nn.Linear(d_model, d_model)  # 512 → 512
  self.linear_k = nn.Linear(d_model, d_model)  # 512 → 512
  self.linear_v = nn.Linear(d_model, d_model)  # 512 → 512
  self.linear_out = nn.Linear(d_model, d_model) # 512 → 512

  输入是 (batch, seq, d_model)，输出还是 (batch, seq, d_model)，维度不变。

  为什么更像 hidden_size

  ┌──────────┬──────────────────────────────┬────────────────────────────────┐
  │ 对比维度 │      RNN 的 hidden_size      │     Transformer 的 d_model     │
  ├──────────┼──────────────────────────────┼────────────────────────────────┤
  │ 定义     │ 隐状态维度，模型内部表示空间 │ 词向量维度，模型的内部表示空间 │
  ├──────────┼──────────────────────────────┼────────────────────────────────┤
  │ 输出形状 │ (batch, seq, hidden_size)    │ (batch, seq, d_model)          │
  ├──────────┼──────────────────────────────┼────────────────────────────────┤
  │ 是否恒定 │ 贯穿所有时间步不变           │ 贯穿所有层不变                 │
  ├──────────┼──────────────────────────────┼────────────────────────────────┤
  │ 决定容量 │ 决定 RNN 的记忆容量          │ 决定 Transformer 的表示容量    │
  └──────────┴──────────────────────────────┴────────────────────────────────┘

  RNN 的 input_size 在 Transformer 中的对应

  Transformer 的"输入维度转换"发生在 Embedding 层，而不是 Encoder 内部：

  # d_model 是 embedding_dim，类比 hidden_size
  embed = MyEmbedding(vocab_size, d_model)  # vocab_size → d_model

  如果硬要类比，RNN 的 input_size 对应的是 Transformer 的 vocab_size（词表大小），而 d_model 对应的是 hidden_size——它定义了模型内部"表示空间"的大小，整个网络都围绕它来构建（QKV 线性层、前馈网络、LayerNorm 都使用 d_model
  作为特征维度）。
'''