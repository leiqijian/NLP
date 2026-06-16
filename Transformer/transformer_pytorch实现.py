"""
演示
    使用nn.Transformer实现一个 完整的transformer模型
组成
    输入部分
    编码器Encoder
    解码器Decoder
    输出部分
需要注意：
    1.nn.Transformer来实现编码器-解码器部分，不包含输入和输出。
    输入和输出需要手动实现：
        - 词嵌入层nn.Embedding
        - 位置编码nn.Embedding
            工程上，采用可学习的位置编码nn.Embedding(max_len,d_model)
            位置编码的最大序列长度采用max_len,是因为：
                1.模型不建议绑定数据参数
                2.支持推理/预测阶段的变长输入
        - 输出线性层nn.Linear(d_model,tgt_vocab_size)
    2.关于掩码mask:
        - 布尔语义:
            True表示被mask的位置,不参与注意力计算;
            False表示没有被mask的位置
        - 统一为2D张量
        - tgt_mask: (tgt_seq_len,tgt_seq_len)
            上三角矩阵，上三角都为True, 对角线及下面为False
            Decoder的掩码自注意力机制的因果掩码，保证Decoder只能看到当前位置以及之前位置，防止看到未来信息
        - src_key_padding_mask: (batch_size,src_seq_len)
            用于Encoder的自注意力机制
            source序列的填充掩码，表示哪些位置被PAD填充，需要进行遮蔽/mask
        - tgt_key_padding_mask: (batch_size,tgt_seq_len)
            用于Decoder的掩码自注意力机制
            target序列的填充掩码，表示哪些位置被PAD填充，需要进行遮蔽/mask
        - memory_key_padding_mask: (batch_size,src_seq_len) = src_key_padding_mask
            用于Decoder的交叉注意力机制
            source序列的填充掩码，表示哪些位置被PAD填充，需要进行遮蔽/mask
    3.参数batch_first = True
        使得所有输入输出的形状为(batch_size, seq_len, *)

"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.定义模型类，实现一个完整的Transformer模型
# 1.1 先定义一个可学习的位置编码类
class LearnablePositionalEncoding(nn.Module):
    # 1.定义__init__方法
    def __init__(self,max_len=1000, d_model=512):
        super().__init__()
        # 1.初始化可学习的位置编码，使用nn.Embedding
        self.position_encoding = nn.Embedding(max_len,d_model) # (max_vocab_size, d_model)
    # 2.forward方法
    def forward(self,x):
        # x: (batch_size, seq_len, d_model), 词向量
        # 1.先获取seq_len
        seq_len = x.shape[1]
        # 2.构造seq_len的索引张量：0~seq_len-1,torch.arange(0,seq_len,)
        position_ids = torch.arange(0,seq_len,dtype=torch.long, device=x.device)
        # (seq_len,) -> (1, seq_len)
        position_ids = position_ids.unsqueeze(0)
        # 3.进行位置编码,(1, seq_len) -> (1, seq_len, d_model) (batch_size, seq_len, d_model)
        pos_encoding = self.position_encoding(position_ids)
        # 4.返回位置编码+词向量x,(batch_size, seq_len, d_model)
        return x + pos_encoding

# 1.2 定义模型类，实现一个完整的transformer模型
class MyTransformer(nn.Module):
    """
    使用nn.Transformer实现一个完整的transformer模型
    参数:
        d_model: 特征维度
        n_heads: 注意力头数
        num_layers: 编码器/解码器层数
        d_ff: 前馈网络层的隐藏层维度
        dropout: 丢弃概率
        batch_first=True: 让输入输出的形状固定为(batch_size,seq_len,*)
        src_vocab_size: 源序列的词表大小
        tgt_vocab_size: 目标序列的词表大小
        max_src_len: 源序列位置编码的最大长度，通常是src_seq_len的2~4倍
        max_tgt_len: 目标序列位置编码的最大长度，通常是tgt_seq_len的2~4倍

    输入:
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        src_key_padding_mask: (batch_size, src_seq_len), 源序列的填充掩码
        tgt_key_padding_mask: (batch_size, tgt_seq_len), 目标序列的填充掩码
        tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
    返回:
        logits: (batch_size, tgt_seq_len, tgt_vocab_size),预测分数
        encoder_output: (batch_size, src_seq_len, d_model), 编码器的输出
        decoder_output: (batch_size, tgt_seq_len, d_model), 解码器的输出
    """
    # 1.定义__init__方法
    def __init__(
            self,
            d_model=512,
            n_heads=8,
            num_layers=6,
            d_ff=2048,
            dropout=0.1,
            batch_first=True,
            src_vocab_size=1000,
            tgt_vocab_size=2000,
            max_src_len=1000,
            max_tgt_len=1000
                 ):
        super().__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.batch_first = batch_first
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        # 2.定义模块
        # 2.1 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)
        # 2.2 可学习的位置编码
        self.src_position_encoding = LearnablePositionalEncoding(max_len=max_src_len, d_model=d_model)
        self.tgt_position_encoding = LearnablePositionalEncoding(max_len=max_src_len, d_model=d_model)
        # 2.3 定义Transformer的编码器-解码器模块
        self.transformer = nn.Transformer(
            d_model=d_model,    # 特征维度
            nhead=n_heads,      # 注意力头数
            num_encoder_layers=num_layers,  # 编码器层数
            num_decoder_layers=num_layers,  # 解码器层数
            dim_feedforward=d_ff,   # 前馈网络层的隐藏层维度
            dropout=dropout,    # 丢弃概率
            batch_first=batch_first,    # 让输入输输出的形状固定为(batch_size,seq_len,*)
            norm_first=True, # 子层连接顺序, True表示pre-LN: LSDA, LayerNorm -> SubLayer -> Add -> Norm; False表示post-LN: SDAL
        )
        # 2.4 线性层, 用于将编码器的输出映射到目标词表大小
        self.fc_out = nn.Linear(d_model,tgt_vocab_size)
    # 2.forward方法
    def forward(
            self,
            src,
            tgt,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            tgt_mask=None):
        # src: (batch_size, src_seq_len),输入的源序列
        # tgt: (batch_size, tgt_seq_len),输入的目标序列
        # src_key_padding_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # tgt_key_padding_mask: (batch_size, tgt_seq_len), 目标序列的填充掩码
        # tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
        # 1.经过词嵌入层
        src_embedding = self.src_embedding(src) # (batch_size, src_seq_len, d_model)
        tgt_embedding = self.tgt_embedding(tgt) # (batch_size, tgt_seq_len, d_model)
        # 2.添加位置编码的词向量
        src_embedding = self.src_position_encoding(src_embedding)   # (batch_size, src_seq_len, d_model)
        tgt_embedding = self.tgt_position_encoding(tgt_embedding)   # (batch_size, tgt_seq_len, d_model)
        # 3.经过Transformer模块,也就是经过编码器-解码器模块
        output = self.transformer(
            src=src_embedding,
            tgt=tgt_embedding,
            src_key_padding_mask=src_key_padding_mask,  # 源序列的填充掩码
            tgt_key_padding_mask=tgt_key_padding_mask,  # 目标序列的填充掩码
            memory_key_padding_mask=src_key_padding_mask,   # 源序列的填充掩码, 用于Decoder的交叉注意力机制
            tgt_mask=tgt_mask,  # 目标序列的因果掩码
        )   # (batch_size, tgt_seq_len, d_model)
        # 4.经过线性层
        logits = self.fc_out(output)    # (batch_size, tgt_seq_len, tgt_vocab_size)
        return logits
    # 3.encode方法,工程上通常也要实现transformer的编码器方法
    def encode(self,src,src_key_padding_mask=None):
        # 1.经过词嵌入层
        src_embedding = self.src_embedding(src)  # (batch_size, src_seq_len, d_model)
        # 2.添加位置编码
        src_embedding = self.src_position_encoding(src_embedding)  # (batch_size, src_seq_len, d_model)
        # 3.经过Transformer.encoder模块,进行编码
        encoder_output = self.transformer.encoder(
            src=src_embedding,
            src_key_padding_mask=src_key_padding_mask,  # 源序列的填充掩码
        )   # (batch_size, src_seq_len, d_model)
        return encoder_output
    # 4.decode方法
    def decode(
        self,tgt,encoder_output,src_key_padding_mask=None,tgt_key_padding_mask=None,tgt_mask=None):
        # tgt: (batch_size, tgt_seq_len),输入的目标序列
        # encoder_output: (batch_size, src_seq_len, d_model), 编码器的输出
        # src_key_padding_mask: (batch_size, src_seq_len), 源序列的填充掩码
        # tgt_key_padding_mask: (batch_size, tgt_seq_len), 目标序列的填充掩码
        # tgt_mask: (tgt_seq_len, tgt_seq_len), 目标序列的因果掩码
        # 1.经过词嵌入层
        tgt_embedding = self.tgt_embedding(tgt)
        # 2.添加位置编码
        tgt_embedding = self.tgt_position_encoding(tgt_embedding)
        # 3.经过Transformer.decoder模块,进行解码
        decoder_output = self.transformer.decoder(
            tgt=tgt_embedding,
            memory=encoder_output,
            tgt_key_padding_mask=tgt_key_padding_mask,  # 目标序列的填充掩码
            memory_key_padding_mask=src_key_padding_mask,   # 源序列的填充掩码, 用于Decoder的交叉注意力机制
            tgt_mask=tgt_mask,  # 目标序列的因果掩码
        )
        return decoder_output

# 定义函数,测试MyTransformer
def test_MyTransformer():
    # 1.初始化参数
    # 模型相关
    d_model = 512
    num_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    src_vocab_size = 2000
    tgt_vocab_size = 4000
    max_src_len = 1000
    max_tgt_len = 1000
    # 数据相关
    batch_size = 4
    src_seq_len = 10    # 源序列的序列长度,也就是截断填充到这个固定长度, 由源序列的句子长度分布情况来选择这个参数
    tgt_seq_len = 20
    # 训练相关
    # epochs = 10
    # lr = 1e-3
    # 2.创建数据
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    print(f"source源序列: {src.shape}")
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    print(f"target目标序列: {tgt.shape}")
    # 3.创建掩码
    # 填充掩码(batch_size, seq_len),这里简化为全是False,也就是没有PAD填充
    src_key_padding_mask = torch.zeros(batch_size, src_seq_len, dtype=torch.bool)
    tgt_key_padding_mask = torch.zeros(batch_size, tgt_seq_len, dtype=torch.bool)
    # 因果掩码(tgt_seq_len, tgt_seq_len),上三角为True,对角线以及下面为False, torch.triu(torch.ones(,),1)
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).type(dtype=torch.bool)
    # 4.创建模型对象
    model = MyTransformer(
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    # 5.前向传播,输出结果
    logits = model(
        src, tgt,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask,
    )   # (batch_size, tgt_seq_len, tgt_vocab_size)
    print(f"logits: {logits.shape}")    # (4,20,4000)
    # 6.测试encode
    encoder_output = model.encode(src, src_key_padding_mask)    # (batch_size, src_seq_len, d_model)
    print(f"encoder_output: {encoder_output.shape}")
    ...

# 测试
if __name__ == '__main__':
    test_MyTransformer()