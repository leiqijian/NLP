
import torch
import torch.nn as nn
import torch.nn.functional as F

# todo: 定义位置编码
class MyPositionalEncoding(nn.Module):
    # 1.定义__init__方法
    def __init__(self,d_model=512,max_len=1000):
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
        # 3.进行位置编码，把位置信息输入到嵌入层中，给位置信息构造成词向量
        # (1, seq_len) -> (1, seq_len, d_model) (batch_size, seq_len, d_model)
        pos_encoding = self.position_encoding(position_ids)
        # 4.返回位置编码+词向量x,(batch_size, seq_len, d_model)
        return x + pos_encoding


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
    def __init__(
            self,
            d_model,
            n_heads,
            d_ff,
            dropout=0.1,
            encoder_num_layers=6,
            decoder_num_layers=6,
            batch_first=True,
            src_vocab_list=None,
            tgt_vocab_list=None,
            max_src_len=1000,
            max_tgt_len=1000
                 ):
        super(MyTransformer, self).__init__()
        # 1.初始化参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.batch_first = batch_first
        self.src_vocab_size = len(src_vocab_list)
        self.tgt_vocab_size = len(tgt_vocab_list)
        # 获取PAD的索引，用于在forward中来构造src_key_padding_mask和tgt_key_padding_mask
        self.src_pad_idx = src_vocab_list.index("<PAD>") if src_vocab_list else 0
        self.tgt_pad_idx = tgt_vocab_list.index("<PAD>") if tgt_vocab_list else 0
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # 2. 定义模块
        # 2.1 定义embedding层,告诉embedding层，pad位置的不需要训练
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model, padding_idx=self.src_pad_idx)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model, padding_idx=self.tgt_pad_idx)
        # 2.2 定义位置编码
        self.src_pos_embedding = MyPositionalEncoding(d_model=self.d_model, max_len = self.max_src_len)
        self.tgt_pos_embedding = MyPositionalEncoding(d_model=self.d_model, max_len = self.max_tgt_len)
        # 2.3 定义transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=encoder_num_layers,
            num_decoder_layers=decoder_num_layers,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=self.batch_first,
            norm_first=True
            )
        # 2.4 定义输出层
        self.fc = nn.Linear(d_model, self.tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_padding_mask,
        tgt_padding_mask,
        tgt_mask
    ):
        src_embedding = self.src_embedding(src)
        tgt_embedding = self.tgt_embedding(tgt)
        src_pos_embedding = self.src_pos_embedding(src_embedding)
        tgt_pos_embedding = self.tgt_pos_embedding(tgt_embedding)
        output = self.transformer(
            src=src_pos_embedding,
            tgt=tgt_pos_embedding,
            src_key_padding_mask=src_padding_mask,  # 源序列的填充掩码
            tgt_key_padding_mask=tgt_padding_mask,  # 目标序列的填充掩码
            memory_key_padding_mask=src_padding_mask,  # 源序列的填充掩码, 用于Decoder的交叉注意力机制
            tgt_mask=tgt_mask,  # 目标序列的因果掩码
        )
        logits = self.fc(output)
        return logits

    def encoder(self, src, src_padding_mask):
        src_embedding = self.src_embedding(src)
        src_pos_embedding = self.src_pos_embedding(src_embedding)
        output = self.transformer.encoder(
            src=src_pos_embedding,
            src_key_padding_mask=src_padding_mask
        )
        return output

    def decoder(self,encoder_output, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        tgt_embedding = self.tgt_embedding(tgt)
        tgt_pos_embedding = self.tgt_pos_embedding(tgt_embedding)
        output = self.transformer.decoder(
            tgt=tgt_pos_embedding,
            memory=encoder_output,
            tgt_key_padding_mask=tgt_key_padding_mask,  # 目标序列的填充掩码
            memory_key_padding_mask=src_key_padding_mask,   # 源序列的填充掩码, 用于Decoder的交叉注意力机制
            tgt_mask=tgt_mask,  # 目标序列的因果掩码
        )
        logits = self.fc(output)
        return logits