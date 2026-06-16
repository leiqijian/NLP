"""
演示
    transformer的 输出部分
组成：
    线性层linear(d_model,tgt_vocab_size)
    softmax(如果使用多分类交叉熵nn.CrossEntropyLoss,就不需要softmax,因为CrossEntropyLoss自带了softmax)
"""
# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_decoder import MyTransformerDecoder

# 1.定义模型类，实现transformer的输出部分
class MyTransformerOutput(nn.Module):
    """
    transformer的输出部分
    输入:
        x: (batch_size, tgt_seq_len, d_model), decoder的输出，预测的token表示
    输出:
        logits: (batch_size, tgt_seq_len, tgt_vocab_size), 预测分数，用CrossEntropyLoss跟真实值算损失，再进行反向传播
        或
        prob: (batch_size, tgt_seq_len, tgt_vocab_size), 预测概率，如果需要预测概率，需要拿到logits之后自己进行一次softmax
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, vocab_size=1000):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.vocab_size = vocab_size #因为目的是要给词表中的所有词都预测一个分数
        # 2.初始化线性层
        self.linear1 = nn.Linear(d_model, vocab_size)
    # 2.forward方法
    def forward(self, x):
        # x: (batch_size, tgt_seq_len, d_model), decoder的输出，预测的token表示
        # 1.经过线性层
        logits = self.linear1(x)    # (batch_size, tgt_seq_len, vocab_size), 预测分数
        # 2.经过softmax得到预测概率(如果使用多分类交叉熵nn.CrossEntropyLoss,就不需要这个softmax)
        # prob = F.softmax(logits, dim=-1)
        return logits

# 2.定义函数，测试MyTransformerOutput
def test_MyTransformerOutput():
    # 1.定义参数
    d_model = 512
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 20
    num_heads = 8
    d_ff = 2048
    vocab_size = 2000
    # 2.创建张量,BSD
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    print(f"输入张量x: {x.shape}")
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
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, src_seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        src_mask[i, :lens[i]] = False
    print(f"填充掩码src_mask: {src_mask.shape}")
    # 因果掩码，tgt_mask: (tgt_seq_len, tgt_seq_len)，
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), 1).to(dtype=torch.bool)
    print(f"因果掩码tgt_mask: {tgt_mask.shape}")

    # 5.前向传播
    output = decoder(
        x,
        encoder_output,
        src_mask,
        tgt_mask
    )
    print(f"输出张量output: {output.shape}")  # (batch_size,tgt_seq_len,d_model)
    # 6.创建MyTransformerOutput对象
    transformer_output = MyTransformerOutput(d_model=d_model, vocab_size=vocab_size)
    # 7.数据传入MyTransformerOutput对象
    logits = transformer_output(output) # (batch_size,tgt_seq_len,vocab_size)
    print(f"预测分数logits: {logits.shape}")
    prob = F.softmax(logits, dim=-1)
    print(f"预测概率prob: {prob.shape}")
    print(f"预测概率prob.sum(dim=-1): {prob.sum(dim=-1)}")

# 测试
if __name__ == '__main__':
    test_MyTransformerOutput()