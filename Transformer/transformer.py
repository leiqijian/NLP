"""
演示
    使用自定义的类，实现完整的transformer模型
组成：
    输入部分
    编码器encoder
    解码器decoder
    输出部分
数据：
    输入数据:
        src: (batch_size, src_seq_len)
        tgt_input: (batch_size, tgt_seq_len)
        src_mask: (batch_size, src_seq_len), 源序列的填充掩码
        tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
    输出数据：
        logits: (batch_size, tgt_seq_len, tgt_vocab_size), 预测分数
        encoder_output: (batch_size, src_seq_len, d_model), 编码器输出
        decoder_output: (batch_size, tgt_seq_len, d_model), 解码器输出
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_decoder import *
from transformer_output import *

# 1.定义模型类，实现完整的transformer模型
class MyTransformer(nn.Module):
    """
    参数:
        d_model: 特征维度
        num_layers:编码器/解码器层数
        num_heads: 注意力头数
        d_ff: 前馈网络的隐藏层维度
        dropout: 随机失活概率
        src_vocab_size: 源序列的词表大小
        tgt_vocab_size: 目标序列的词表大小
        max_len: 位置编码的最大序列长度
    输入:
        src: (batch_size, src_seq_len)
        tgt_input: (batch_size, tgt_seq_len)
        src_mask: (batch_size, src_seq_len), 源序列的填充掩码
        tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
    输出:
        logits: (batch_size, tgt_seq_len, tgt_vocab_size), 预测分数
        encoder_output: (batch_size, src_seq_len, d_model), 编码器输出
        decoder_output: (batch_size, tgt_seq_len, d_model), 解码器输出
    """
    # 1.初始化__init__方法
    def __init__(
        self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, num_layers=6, src_vocab_size=1000, tgt_vocab_size=2000, max_len=10000):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.num_heads = num_heads
        # 添加整除校验逻辑，d_model % num_heads == 0
        assert d_model % num_heads == 0, f"d_model={d_model} % num_heads={num_heads} != 0"
        self.d_ff = d_ff
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        # 2.定义模块
        # 输入部分
        # src输入
        self.src_embed = nn.Sequential(
            MyEmbedding(vocab_size=src_vocab_size,d_model=d_model),
            MyPositionalEncoding(d_model=d_model,dropout=dropout,max_len=max_len)
        )
        # tgt输入
        self.tgt_embed = nn.Sequential(
            MyEmbedding(vocab_size=tgt_vocab_size,d_model=d_model),
            MyPositionalEncoding(d_model=d_model,dropout=dropout,max_len=max_len)
        )
        # 编码器
        self.encoder = MyTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=num_layers
        )
        # 解码器
        self.decoder = MyTransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=num_layers
        )
        # 输出部分
        self.output = MyTransformerOutput(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        )

    # 2.forward方法
    def forward(self, src, tgt_input, src_mask=None, tgt_mask=None):
        # src: (batch_size, src_seq_len)
        # tgt_input: (batch_size, tgt_seq_len)
        # src_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
        # 1.经过输入部分
        src_embed = self.src_embed(src) # (batch_size, src_seq_len, d_model)
        tgt_embed = self.tgt_embed(tgt_input)   # (batch_size, tgt_seq_len, d_model)
        # 2.经过编码器
        encoder_output = self.encoder(x=src_embed, padding_mask=src_mask)
        # 3.经过解码器x, encoder_output, src_mask=None, tgt_mask=None
        decoder_output = self.decoder(
            x=tgt_embed,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        # 4.经过输出部分
        logits = self.output(decoder_output)
        return logits
    # 3.encode方法
    def encode(self, src, src_mask=None):
        # src: (batch_size, src_seq_len)
        # src_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # 1.经过输入部分
        src_embed = self.src_embed(src)  # (batch_size, src_seq_len, d_model)
        # 2.经过编码器
        encoder_output = self.encoder(x=src_embed, padding_mask=src_mask)
        return encoder_output
    # 4.decode方法
    def decode(self, tgt_input, encoder_output, src_mask=None, tgt_mask=None):
        # tgt_input: (batch_size, tgt_seq_len)
        # encoder_output: (batch_size, src_seq_len, d_model), 编码器的输出
        # src_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
        # 1.经过输入部分
        tgt_embed = self.tgt_embed(tgt_input)  # (batch_size, tgt_seq_len, d_model)
        # 2.经过解码器
        decoder_output = self.decoder(
            x=tgt_embed,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        return decoder_output

# 2.定义函数，测试MyTransformer模型
def test_MyTransformer():
    # 1.定义参数
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 20
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    src_vocab_size = 1000
    tgt_vocab_size = 2000
    dropout = 0.1
    max_len = 5000
    # 2.创建张量
    # src: (batch_size, src_seq_len)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    # tgt_input: (batch_size, tgt_seq_len)
    tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    # 3.创建掩码矩阵
    # 源序列的填充掩码，src_mask: (batch_size, src_seq_len)
    src_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool)
    # print(src_mask)
    # 随机生成填充掩码的有效长度，有效长度内都设置为False
    lens = torch.randint(5, src_seq_len + 1, (batch_size,))
    # for循环来实现有效长度内的填充掩码都设置为False,padding_mask[i,:lens[i]]=False
    for i in range(batch_size):
        src_mask[i, :lens[i]] = False
    # print(f"填充掩码src_mask: {src_mask}")
    # 因果掩码，tgt_mask: (tgt_seq_len, tgt_seq_len)，
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), 1).to(dtype=torch.bool)
    # print(f"因果掩码tgt_mask: {tgt_mask}")
    # 4.创建MyTransformer模型对象
    model = MyTransformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout=dropout,
        max_len=max_len
    )
    # 5.前向传播 src, tgt_input, src_mask=None, tgt_mask=None
    logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"logits: {logits.shape}")    # (batch_size, tgt_seq_len, tgt_vocab_size)
    # 6.测试encode
    encoder_output = model.encode(src, src_mask)
    print(f"encoder_output: {encoder_output.shape}")
    # 7.测试decode
    decoder_output = model.decode(tgt_input, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"decoder_output: {decoder_output.shape}")
    ...

# 测试
if __name__ == '__main__':
    test_MyTransformer()