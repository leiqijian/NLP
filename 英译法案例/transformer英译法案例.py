"""
演示:
    Transformer英译法案例，基于 Transformer模型，实现英译法任务，属于seq2seq的经典案例。
模型结构：
    输入部分：
        词嵌入层nn.Embedding(vocab_size, d_model)
        可学习位置编码nn.Embedding(max_len, d_model)
    EncoderDecoder:
        nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    输出部分：
        线性层nn.Linear(d_model, vocab_size)
输入输出数据：
    源序列source：(batch_size, src_seq_len)
    目标序列target: (batch_size, tgt_seq_len)
    输入:
        源序列source: (batch_size, src_seq_len)
        目标输入tgt_input: target[:,:-1], (batch_size, tgt_seq_len-1)
    输出:
        预测输出logits：(batch_size, tgt_seq_len-1, tgt_vocab_size)
        预测结果y_pred: (batch_size, tgt_seq_len-1)
        真实输出tgt_output: target[:,1:],(batch_size, tgt_seq_len-1)
掩码mask：
    - 源序列填充掩码src_key_padding_mask：(batch_size, src_seq_len),表示源序列中哪些位置是<PAD>填充
    - 目标序列填充掩码tgt_key_padding_mask：(batch_size, tgt_seq_len-1),表示目标序列中哪些位置是<PAD>填充
    - 目标序列因果掩码tgt_mask：(tgt_seq_len-1, tgt_seq_len-1),表示目标序列中每个位置能看到哪些位置，被mask的位置为True
案例的工作流：
    1.准备数据-重点
        1.清洗文本，去掉无效字符
        2.构造source-target句子对，拆分数据集为 train/val/test
        3.构建词表，仅使用训练集，输入词表src_vocab，输出词表tgt_vocab
        4.查看句子长度分布，选择句子最大长度seq_len
        5.构建自定义数据集(截断填充 source target 到 固定长度)

    2.搭建神经网络-重点
        输入部分 + EncoderDecoder + 输出部分
    3.模型训练-重点
        在训练集上训练，在验证集上验证，在测试集上测试
        准确率accuracy为token级别的准确率,忽略<PAD>的影响
        teacher forcing, 是否使用真实标签预测下个token
    4.模型测试
        teacher forcing测试
        自回归测试(工程主流)
    5.模型预测-自回归推理
        输入一个英文句子，输出翻译的法语句子
关键点说明:
    1.source和target都要截断填充到最大长度SRC_SEQ_LEN/TGT_SEQ_LEN, 首尾要添加特殊表示，开始<SOS>,结束<EOS>,填充<PAD>,未知词用<UNK>表示(词表中没有的词,
    OOV:out-of-vocab)
    2.词表中要包含特殊符号: <PAD>,<SOS>,<EOS>,<UNK>, <PAD>是填充符号，通常在embedding中设置padding_idx = 0(PAD填充的词表索引)
    3.位置编码使用可学习的位置编码，位置编码的序列长度一般为seq_len的2~4倍，主要是考虑：1.模型中一般不会直接定义数据参数；2.支持推理/预测阶段的变长输入
    4.损失和准确率都是按照token计算的token级别的指标，忽略<PAD>; 实际机器翻译项目中使用BLEU指标

"""

# 导包
import json # 保存词表为json文件
import os  # 文件操作
import re  # 用于正则表达式
import time
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# 绘图
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # 用于拆分数据集
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 苹果电脑
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 0.全局配置与常量
# 设置设备，CUDA>MPS>CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"当前设备: {DEVICE}")
# 特殊token
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# 文件路径
DATA_PATH = r"./data/eng-fra-v2.txt"
MODEL_DIR = r"./model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "transformer_en2fr.pt") # pytorch模型文件格式, .pt/.pth
IMAGE_DIR = r"./image"
os.makedirs(IMAGE_DIR, exist_ok=True)
IMAGE_FILE = os.path.join(IMAGE_DIR, "train_loss_acc.png")

# 词表路径
SRC_VOCAB_PATH = os.path.join(MODEL_DIR, "src_vocab.json")
TGT_VOCAB_PATH = os.path.join(MODEL_DIR, "tgt_vocab.json")

# 超参数
# 模型相关
D_MODEL = 512
N_HEAD = 8
NUM_LAYERS = 2
D_FF = D_MODEL*4
DROPOUT = 0.1

# 训练相关
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-3 # 权重衰减系数

# 数据相关
BATCH_SIZE = 64
SRC_SEQ_LEN = 12
TGT_SEQ_LEN = 12
# 设置max_len,一般为seq_len的2~4倍
MAX_SRC_LEN = 32
MAX_TGT_LEN = 32

# todo 1.准备数据-重点
# 1.清洗文本，去掉无效字符
def clean_text(text):
    # 1.转为小写，去除首尾的空白符号
    string = text.lower().strip()

    # 2.在标点.!?前后添加空格
    string = re.sub(r"([.!?])", r" \1 ", string)

    # 3.把特殊符号(大小写字母 和 .!?之外的符号)都替换成空格
    string = re.sub(r"[^a-zA-Z.!?]+", " ", string)
    return string.strip()

# 2.构造source-target句子对，拆分数据集为 train/val/test
def split_data(data_path=DATA_PATH, train_ratio=0.8, val_ratio=0.1):
    # 1.读取txt文件
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 2.切分src-tgt句子对,构造src-tgt句子对列表
    pairs = [line.strip().split("\t") for line in lines]
    # 对pairs中的src, tgt进行文本清洗clean_text
    pairs = [(clean_text(src),clean_text(tgt)) for src,tgt in pairs]

    # 3.去重,一定要去重，否则会造成数据泄露
    # dict.fromkeys()
    pairs = list(dict.fromkeys(pairs))

    # 4.拆分数据集为 train/val/test
    train_pairs, valid_test_pairs = train_test_split(pairs, train_size=train_ratio, random_state=7)
    valid_pairs, test_pairs = train_test_split(valid_test_pairs, train_size=val_ratio/(1-train_ratio), random_state=7)

    return train_pairs, valid_pairs, test_pairs

# 3.构建词表，仅使用训练集，输入词表src_vocab，输出词表tgt_vocab
def build_vocab(pairs):
    # 1.添加特殊token
    src_vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    tgt_vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    # 2.构建词表，token:id 的映射字典
    for src, tgt in pairs:
        # 1.构建 src_vocab
        # 遍历src.split(),默认用空白符号(" ","  ","\t","\n")进行切分,获取每个单词
        for token in src.split():
            if token not in src_vocab:
                src_vocab[token] = len(src_vocab)
        # 2.构建 tgt_vocab
        for token in tgt.split():
            if token not in tgt_vocab:
                tgt_vocab[token] = len(tgt_vocab)

    # 3.构建反义词表，id:token 的映射字典
    rev_src_vocab = {id: token for token, id in src_vocab.items()}
    rev_tgt_vocab = {id: token for token, id in tgt_vocab.items()}
    return src_vocab, tgt_vocab, rev_src_vocab, rev_tgt_vocab

# 保存词表为json文件 - 工程化处理
def save_vocab(vocab, vocab_path):
    """
    保存词表到JSON文件
    :param vocab: 词表
    :param vocab_path: 词表保存路径
    """
    with open(vocab_path, "w", encoding="utf-8") as f:
        # 将词表字典序列化为JSON格式并写入文件
        # ensure_ascii=False: 允许非ASCII字符（如中文）直接显示，而不是转义为\uXXXX
        # indent=4: 设置缩进为4个空格，使生成的JSON文件更易读
        json.dump(vocab, f, ensure_ascii=False, indent=4)

# 加载词表 - 工程化处理
def load_vocab(vocab_path):
    """
    从JSON文件加载词表
    :param vocab_path: 词表文件路径
    :return: 加载的词表字典
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

# 4.查看句子长度分布，选择句子最大长度seq_len
def plot_seq_len_dist(pairs):
    # 1.统计 source 和 target 的句子长度
    src_lens = [len(source.split()) for source, target in pairs]
    tgt_lens = [len(target.split()) for source, target in pairs]
    # 2.创建1*2子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # 3.绘制source / target 的句子长度分布直方图
    for ax, lens, name in zip(axs, [src_lens, tgt_lens], ["source句子长度分布", "target句子长度分布"]):
        sns.histplot(lens, ax=ax, bins=10, kde=False)
        ax.set_title(name)
        ax.set_xlabel("句子长度")
        ax.set_ylabel("句子数量")
    plt.tight_layout()
    plt.show()

# 5.构建自定义数据集(截断填充 source target 到 固定长度)- 要求手敲
# DataSet: 定义一个样本是什么，这里定义一个样本的source-target
class MyDataset(Dataset):
    """
    自定义的数据集，对输入的样本src/tgt截断填充到固定长度seq_len,构造为标准src-tgt样本
    """
    # 1.定义__init__方法
    def __init__(self, pairs, src_vocab, tgt_vocab):
        super().__init__()
        # 1.初始化参数
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_seq_len = SRC_SEQ_LEN
        self.tgt_seq_len = TGT_SEQ_LEN
        self.num_samples = len(pairs)

    # 2.定义__len__方法，返回数据集样本数量
    def __len__(self):
        return self.num_samples

    # 3.定义__getitem__方法，返回指定索引的样本的source,target
    def __getitem__(self, idx):
        # 1.获取当前索引对应的src, tgt
        src, tgt = self.pairs[idx]

        # 2.把文本序列转为 索引序列
        src_index = [self.src_vocab.get(token, self.src_vocab.get("<UNK>",3)) for token in src.split()]
        tgt_index = [self.tgt_vocab.get(token, self.tgt_vocab.get("<UNK>",3)) for token in tgt.split()]

        # 3.截断填充句子到固定长度seq_len
        src_index = self.pad_seq(src_index, self.src_vocab, self.src_seq_len)
        tgt_index = self.pad_seq(tgt_index, self.tgt_vocab, self.tgt_seq_len)
        return src_index, tgt_index

    # 4.定义pad_seq方法，截断填充句子到固定长度seq_len
    def pad_seq(self, seq, vocab, seq_len):
        # 1.截断序列长度为seq_len-2
        seq = seq[:seq_len-2]

        # 2.开始处添加<SOS>,结尾处添加<EOS>
        seq = [vocab.get("<SOS>", 1)] + seq + [vocab.get("<EOS>", 2)]

        # 3.填充<PAD>到seq_len
        seq = seq + [vocab.get("<PAD>", 0)] * (seq_len - len(seq))

        # 4.返回张量
        return torch.tensor(seq, dtype=torch.int64)


# todo 2.搭建神经网络-重点
# 1.构建可学习的位置嵌入层，nn.Embedding(max_len,d_model)
class LearnablePositionalEmbedding(nn.Module):
    """
    实现一个可学习的位置编码，使用nn.Embedding实现
    输入:
        x: BSD,词向量
    输出:
        x: BSD, 添加位置编码的词向量
    """
    # 1.定义__init__方法
    def __init__(self, d_model=512, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 1.创建位置嵌入层
        self.pos_embedding = nn.Embedding(max_len, d_model)

    # 2.定义forward方法
    def forward(self, x):
        # x: BSD,词向量
        # 1.构造 seq_len 方向的索引 0~seq_len-1
        seq_len = x.size(1)
        # 1D:0~seq_len-1,(seq_len)
        # 把pos_ids升维到2D:BS,(seq_len,)->(1,seq_len)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        # 2.进行位置编码，把seq_len索引 输入到位置嵌入层
        pos_embedding = self.pos_embedding(pos_ids) # (1,seq_len,d_model)

        # 3.返回 词向量 + 位置编码
        # BSD + 1SD -> BSD
        return x + pos_embedding

# 2.定义模型类，实现一个完整的Transformer模型 - 要求手敲
# 输入部分 + EncoderDecoder + 输出部分
class MyTransformer(nn.Module):
    """
    自定义的Transformer模型，用于英译法任务，seq2seq任务
    组成：输入部分(src/tgt词嵌入层 + src/tgt位置嵌入层) + EncoderDecoder + 输出部分
    参数:
        src_vocab: 源文本词表
        tgt_vocab: 目标文本词表
        d_model: 特征维度
        n_heads: 多头注意力的头数
        num_layers: 编码器/解码器的层数
        d_ff: 隐藏层维度
        dropout: 随机失活概率
    输入:
        source源序列: (batch_size, src_seq_len)
        tgt_input目标输入: target[:,:-1], (batch_size, tgt_seq_len-1)
        src_key_padding_mask源序列填充掩码：(batch_size, src_seq_len),表示源序列中哪些位置是<PAD>填充
        tgt_key_padding_mask目标序列填充掩码：(batch_size, tgt_seq_len-1),表示目标序列中哪些位置是<PAD>填充
        tgt_mask目标序列因果掩码：(tgt_seq_len-1, tgt_seq_len-1),表示目标序列中每个位置能看到哪些位置，被mask的位置为True
    输出:
        logits预测输出：(batch_size, tgt_seq_len-1, tgt_vocab_size)
        y_pred预测结果: (batch_size, tgt_seq_len-1), logits.argmax(dim=-1)
        tgt_output真实目标值: target[:,1:],(batch_size, tgt_seq_len-1)
        encoder_output编码器输出: BSD(batch_size, src_seq_len, d_model)
        decoder_output解码器输出: BSD(batch_size, tgt_seq_len-1, d_model)
    """
    # 1.定义__init__方法
    def __init__(
        self, src_vocab, tgt_vocab,
        d_model=512, n_heads=8, num_layers=6, d_ff=2048, dropout=0.1,
    ):
        super().__init__()
        # 1.初始化参数
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        # 获取PAD的索引，用于在forward中来构造填充掩码:src_key_padding_mask, tgt_key_padding_mask
        self.src_pad_idx = self.src_vocab.get("<PAD>", 0)
        self.tgt_pad_idx = self.tgt_vocab.get("<PAD>", 0)

        # 2.创建模块
        # 2.1 词嵌入层 + 位置嵌入
        # padding_idx: 填充索引，告诉词嵌入层，填充索引的词向量不需要训练
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=self.src_pad_idx)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=self.tgt_pad_idx)
        # 位置嵌入
        self.src_pos_embedding = LearnablePositionalEmbedding(d_model=d_model, max_len=MAX_SRC_LEN)
        self.tgt_pos_embedding = LearnablePositionalEmbedding(d_model=d_model, max_len=MAX_TGT_LEN)

        # 2.2 EncoderDecoder模块，使用nn.Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, # 特征维度
            nhead=n_heads, # 注意力头数
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff, #  隐藏层维度
            dropout=dropout,
            batch_first=True, # 让输入输出数据的形状固定为(B,S,*)
            norm_first=True, # 子层连接顺序，True表示pre-LN: LSDA(LayerNorm-Sublayer-Dropout-Add);False表示post-LN:SDAL
        )

        # 2.3 输出线性层
        self.linear1 = nn.Linear(d_model, len(tgt_vocab))

    # 构造因果掩码的方法,私有方法，只能内部调用
    def _get_causal_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool, device=DEVICE)

    # 2.定义forward方法
    def forward(self, source, target=None, is_inference=False):
        """
        根据target来判断训练/预测，如果target不为None，则为训练模式，否则为预测/推理模式
        需要构造输入输出数据:
            输入数据:
                source (batch_size, src_seq_len)
                tgt_input = target[:, :-1] (batch_size, tgt_seq_len - 1)
            输出数据(这里无需考虑):
                真实目标值 tgt_output = target[:, 1:]

        在forward内部构造掩码，是因为src/tgt包含了构造掩码需要的信息。
        - src_key_padding_mask源序列填充掩码：(batch_size, src_seq_len), 表示源序列中哪些位置是 < PAD > 填充
        - tgt_key_padding_mask目标序列填充掩码：(batch_size, tgt_seq_len - 1), 表示目标序列中哪些位置是 < PAD > 填充
        - tgt_mask目标序列因果掩码：(tgt_seq_len - 1,tgt_seq_len - 1), 表示目标序列中每个位置能看到哪些位置，被mask的位置为True
        过程：
        0.根据target是否为空，来决定是训练模式 or 推理模式
        1.训练模式(target is not None)
            1.构造source, tgt_input
            2.构造掩码
            3.前向传播
        2.推理模式(target is None)
            模型逐个预测下一个token，然后把新预测的token拼接回预测序列generated_tokens中,继续预测下一个token
            1.计算自回归生成的最大长度max_len
            2.构造源文本的填充掩码src_key_padding_mask
            3.获取编码器输出encoder_output
            4.初始化生成序列generated_tokens
            5.自回归推理，逐个生成下个token

        :param source: 源序列，BS(batch_size, src_seq_len)
        :param target: 目标序列，BS(batch_size, tgt_seq_len)
        :param is_inference: True为最终推理(序列最大长度为MAX_TGT_LEN)；False为测试阶段(序列最大长度为TGT_SEQ_LEN)
        :return:
            训练模式：{"logits": logits}，logits: (batch_size, tgt_seq_len-1, tgt_vocab_size)，预测结果
            推理模式：{"logits": all_logits, "generated_tokens": generated_tokens}，包含预测分数和生成的 token 序列
        """
        # 0.根据target是否为空，来决定是训练模式 or 推理模式
        # 1.训练模式(target is not None)
        if target is not None:
            # 1.构造source, tgt_input
            # source = source
            tgt_input = target[:,:-1]   # BS(batch_size, tgt_seq_len - 1)

            # 2.构造掩码
            # 填充掩码: BS
            src_key_padding_mask = (source == self.src_pad_idx).to(DEVICE) # BS(batch_size, src_seq_len)
            tgt_key_padding_mask = (tgt_input == self.tgt_pad_idx).to(DEVICE) # BS(batch_size, tgt_seq_len - 1)
            # 因果掩码: SS
            tgt_mask = self._get_causal_mask(seq_len=tgt_input.size(1)) # SS(tgt_seq_len - 1,tgt_seq_len - 1)

            # 3.前向传播
            # 词嵌入层 + 位置编码
            src_embedding = self.src_embedding(source) # BSD
            src_embedding = self.src_pos_embedding(src_embedding) # BSD
            tgt_embedding = self.tgt_embedding(tgt_input)
            tgt_embedding = self.tgt_pos_embedding(tgt_embedding)

            # 编码器解码器模块
            decoder_output = self.transformer(
                src = src_embedding, # BSD, 源序列的词向量,
                tgt=tgt_embedding, # BSD, 目标序列的词向量
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask, # 因果掩码，SS(tgt_seq_len - 1,tgt_seq_len - 1)
            ) # BSD

            # 线性层
            logits = self.linear1(decoder_output) # BSV(B, tgt_seq_len-1, tgt_vocab_size)

            return {"logits": logits}

        # 2.推理模式(target is None)：自回归测试过程/自回归推理过程
        else:
            # 模型逐个预测下一个token，然后把新预测的token拼接回预测序列generated_tokens中,继续预测下一个token
            # 思路：初始化generated_tokens: <SOS> -> 预测第一个词 token1 -> 拼接generated_tokens: <SOS> token1
            # -> 预测第2个词 token2 -> 拼接generated_tokens: <SOS> token1 token2
            # 直到最大长度 或者遇到<EOS>，停止生成
            # 1.计算自回归生成的最大长度max_len
            # 开头为<SOS>,所以要 -1
            # 如果is_inference=True,那么是纯推理过程，支持长文本输入，所以max_len = MAX_TGT_LEN
            if is_inference:
                max_len = MAX_TGT_LEN - 1
            # 如果is_inference=False,那么是自回归测试过程，支持长文本输入，所以max_len = TGT_SEQ_LEN
            else:
                max_len = TGT_SEQ_LEN - 1
            # 2.构造源文本的填充掩码src_key_padding_mask
            # src_key_padding_mask: BS(batch_size, src_seq_len)
            src_key_padding_mask = (source == self.src_pad_idx).to(DEVICE)  # BS(batch_size, src_seq_len)

            # 3.获取编码器输出encoder_output
            encoder_output = self.encode(source, src_key_padding_mask) # BSD

            # 4.初始化生成序列generated_tokens
            # generated_tokens: B1(batch_size,1)
            # <SOS>
            generated_tokens = torch.full((source.shape[0],1), self.tgt_vocab.get("<SOS>",1)).to(dtype=torch.long, device=DEVICE)

            # 5.自回归推理，逐个生成下个token
            # 初始化预测列表logits: BSV, [BV,BV,...]
            all_logits = []
            for i in range(max_len):
                # 0.由当前生成序列generated_tokens生成tgt_input
                # Bt:(batch_size, i+1)
                tgt_input = generated_tokens

                # 1.构造掩码
                # tgt_key_padding_mask: Bt:(batch_size, i+1)
                tgt_key_padding_mask = (tgt_input == self.tgt_pad_idx).to(DEVICE) # Bt:(batch_size, i+1)

                # tgt_mask: tt
                tgt_mask = self._get_causal_mask(seq_len=i+1)

                # 2.进行decode，得到decoder_output
                decoder_output = self.decode(
                    encoder_output, tgt_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask
                ) # BtD:(batch_size, i+1, tgt_vocab_size)

                # 3.经过线性层
                # 最后一个token的表示decoder_output[:,-1,:]: BD -> linear -> logits
                logits = self.linear1(decoder_output[:,-1,:]) # BV

                # 4.保存当前logits到列表all_logits中
                all_logits.append(logits) # [BV,BV,...]

                # 5.获取当前生成的token: next_token
                next_token = logits.argmax(dim=-1) # B

                # 6.保存当前token到生成序列generated_tokens中
                # generated_tokens + next_token : Bt + B1 -> (B,t+1)
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=-1) # (B,t+1)

            # 返回结果
            return {
                "logits": torch.stack(all_logits, dim=1), # BSV(batch_size, max_len, vocab_size)
                "generated_tokens": generated_tokens[:,1:], # BS (batch_size, max_len) <SOS>
            }

    # 3.定义encode方法
    def encode(self, source, src_key_padding_mask=None):
        # 1.经过词嵌入层 + 位置嵌入层
        src_embedding = self.src_embedding(source)  # BSD
        src_embedding = self.src_pos_embedding(src_embedding)  # BSD

        # 2.经过Transformer.encoder
        encoder_output = self.transformer.encoder(
            src=src_embedding,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoder_output

    # 4.定义decode方法
    def decode(
        self, encoder_output, tgt_input,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None
    ):
        # 1.经过词嵌入层 + 位置嵌入层
        tgt_embedding = self.tgt_embedding(tgt_input)
        tgt_embedding = self.tgt_pos_embedding(tgt_embedding) # BSD

        # 2.经过Transformer.decoder
        decoder_output = self.transformer.decoder(
            tgt = tgt_embedding,
            memory = encoder_output,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        ) # BSD
        return decoder_output

# todo 3.模型训练-重点
# 在训练集上训练，在验证集上验证，在测试集上测试
# 准确率accuracy为token级别的准确率,忽略<PAD>的影响
# teacher forcing, 是否使用真实标签预测下个token
# 3.1 train_one_epoch - 要求手敲
def train_one_epoch(model, train_dataloader, loss_fn, optimizer):
    """
    训练一个轮次，计算loss，acc, 按照token来计算的
    :param model: 模型对象
    :param train_dataloader: 训练集的数据加载器
    :param loss_fn: 损失函数对象
    :param optimizer: 优化器对象
    :return: 当前epoch的平均损失和准确率
    """
    # 1.设置为训练模式
    model.train()

    # 2.初始化当前轮次的总损失、预测正确的token数量、总的有效token数量，按照token来计算
    total_loss = 0.0
    total_correct_tokens = 0
    total_valid_tokens = 0

    # 3.遍历数据加载器，分批次训练
    for src, tgt in tqdm(train_dataloader, desc="Training"):
        # src: BS(batch_size, src_seq_len)
        # tgt: BS(batch_size, tgt_seq_len)
        # 0.迁移数据到设备
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # 1.前向传播
        y_output = model(source=src, target=tgt)
        logits = y_output["logits"] # BSV(B, tgt_seq_len-1, tgt_vocab_size)
        # 构造真实目标值tgt_output
        tgt_output = tgt[:,1:]
        # print(f"tgt: {tgt.shape}")
        # print(f"tgt_output: {tgt_output.shape}")
        # print(f"tgt_output: \n{tgt_output}")

        # 2.计算损失
        # loss=loss_fn(logits:2D, y_true:1D)
        logits_2d = logits.reshape(-1, logits.shape[-1])
        tgt_output_1d = tgt_output.reshape(-1)
        loss = loss_fn(logits_2d, tgt_output_1d)
        # print(f"tgt_output_1d: \n{tgt_output_1d}")

        # 3.梯度清零
        optimizer.zero_grad()
        # 4.反向传播
        loss.backward()
        # 5.更新参数
        optimizer.step()

        # 6.更新指标
        # 计算有效token的数量，非PAD的token
        # tgt填充mask,代表tgt_output中哪些位置是PAD
        mask = tgt_output == model.tgt_pad_idx
        # 统计非PAD位置的总数量，也就是有效token的数量
        num_tokens = torch.sum(~mask).item()
        # 更新总损失
        total_loss += loss.item() * num_tokens
        # 总的有效token数量
        total_valid_tokens += num_tokens
        # 预测正确的token数量
        total_correct_tokens += torch.sum((logits.argmax(dim=-1) == tgt_output)&(~mask)).item()

    # 4.计算平均损失、准确率
    avg_loss = total_loss / total_valid_tokens
    acc = total_correct_tokens / total_valid_tokens
    # 5.返回
    return avg_loss, acc

# 3.2 evaluate
@torch.no_grad()
def evaluate(model, valid_dataloader, loss_fn, teacher_forcing=True):
    """
    评估一个轮次，计算loss，acc, 按照token来计算的
    :param model: 模型对象
    :param valid_dataloader: 验证集的数据加载器
    :param loss_fn: 损失函数对象
    :param teacher_forcing: 是否使用真实标签预测下个token
    :return: 当前epoch的平均损失和准确率
    """
    # 1.设置为评估模式
    model.eval()

    # 2.初始化当前轮次的总损失、预测正确的token数量、总的有效token数量，按照token来计算
    total_loss = 0.0
    total_correct_tokens = 0
    total_valid_tokens = 0

    # 3.遍历数据加载器，分批次评估
    for src, tgt in tqdm(valid_dataloader, desc="Evaluating"):
        # src: BS(batch_size, src_seq_len)
        # tgt: BS(batch_size, tgt_seq_len)
        # 0.迁移数据到设备
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # 1.前向传播
        if teacher_forcing:
            y_output = model(source=src, target=tgt)
        else:
            y_output = model(source=src, target=None)

        logits = y_output["logits"] # BSV(B, tgt_seq_len-1, tgt_vocab_size)
        # 构造真实目标值tgt_output
        tgt_output = tgt[:,1:]
        # print(f"tgt: {tgt.shape}")
        # print(f"tgt_output: {tgt_output.shape}")
        # print(f"tgt_output: \n{tgt_output}")

        # 2.计算损失
        # loss=loss_fn(logits:2D, y_true:1D)
        logits_2d = logits.reshape(-1, logits.shape[-1])
        tgt_output_1d = tgt_output.reshape(-1)
        loss = loss_fn(logits_2d, tgt_output_1d)
        # print(f"tgt_output_1d: \n{tgt_output_1d}")

        # # 3.梯度清零
        # optimizer.zero_grad()
        # # 4.反向传播
        # loss.backward()
        # # 5.更新参数
        # optimizer.step()

        # 6.更新指标
        # 计算有效token的数量，非PAD的token
        # tgt填充mask,代表tgt_output中哪些位置是PAD
        mask = tgt_output == model.tgt_pad_idx
        # 统计非PAD位置的总数量，也就是有效token的数量
        num_tokens = torch.sum(~mask).item()
        # 更新总损失
        total_loss += loss.item() * num_tokens
        # 总的有效token数量
        total_valid_tokens += num_tokens
        # 预测正确的token数量
        total_correct_tokens += torch.sum((logits.argmax(dim=-1) == tgt_output)&(~mask)).item()

    # 4.计算平均损失、准确率
    avg_loss = total_loss / total_valid_tokens
    acc = total_correct_tokens / total_valid_tokens
    # 5.返回
    return avg_loss, acc

# 3.3 train 训练主函数 - 要求手敲
def train(
    train_dataset, valid_dataset, src_vocab, tgt_vocab
):
    """
    训练主函数
        在训练集上训练，在验证集上验证，在测试集上测试
        准确率accuracy为token级别的准确率,忽略<PAD>的影响
        teacher forcing, 是否使用真实标签预测下个token
    :param train_dataset: 训练集
    :param valid_dataset: 验证集
    :param src_vocab: 源文本的词表
    :param tgt_vocab: 目标文本的词表
    :return: 训练/验证的 损失列表 + 准确率列表
    """
    # 1.准备数据集
    # DataSet -> DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # 训练时打乱数据，防止模型学到批次划分的无用信息
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 验证集不进行打乱
    )

    # 2.创建模型对象
    model = MyTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        n_heads=N_HEAD,
        dropout=DROPOUT,
    ).to(DEVICE)

    # 3.创建损失函数和优化器
    # ignore_index: 计算损失时，要忽略的token索引, 一般为PAD填充的索引
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.get("<PAD>",0))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4.初始化训练指标，训练损失列表、训练准确率列表、验证损失列表、验证准确率列表
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    # 初始化最优验证损失
    # 到底有哪个验证指标来判断最优模型，取决于业务需求，这里直接用验证损失
    best_valid_loss = float("inf")

    # 5.开始训练，遍历epochs
    for epoch in range(EPOCHS):
        # 1.在训练集上训练一个轮次，得到训练损失、训练准确率
        train_loss, train_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer)

        # 2.在验证集上评估，得到验证损失、验证准确率
        valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn)

        # 3.根据验证损失来保存模型
        if valid_loss < best_valid_loss:
            # 1.更新最优验证损失
            best_valid_loss = valid_loss
            # 2.保存模型参数
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"保存最优模型:{MODEL_FILE}，当前验证损失：{valid_loss:.4f}")

        # 4.记录 训练/验证的 损失、准确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # 5.打印训练指标
        print(f"Epoch: {epoch+1}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"valid_loss: {valid_loss:.4f} | "
              f"valid_acc: {valid_acc:.4f}")

    # 6.返回结果
    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "valid_losses": valid_losses,
        "valid_accs": valid_accs,
    }

# 3.4 绘图函数（损失与准确率曲线）
def plot_history(history, out_path=IMAGE_FILE):
    """
    绘制训练/验证的 loss 与 accuracy 曲线（两幅子图）
    """
    epochs = len(history["train_losses"])  # 总 epoch 数
    x = list(range(1, epochs + 1))  # 横轴
    plt.figure(figsize=(10, 5))  # 画布大小
    # subplot 1: loss
    plt.subplot(1, 2, 1)  # 左子图
    plt.plot(x, history["train_losses"], label="train_loss")  # 训练损失曲线
    plt.plot(x, history["valid_losses"], label="val_loss")  # 验证损失曲线
    plt.xlabel("epoch")  # 横轴标题
    plt.ylabel("loss")  # 纵轴标题
    plt.title("token-level Loss")  # 子图标题
    plt.grid()
    plt.legend()  # 图例
    # subplot 2: accuracy
    plt.subplot(1, 2, 2)  # 右子图
    plt.plot(x, history["train_accs"], label="train_acc")  # 训练准确率
    plt.plot(x, history["valid_accs"], label="val_acc")  # 验证准确率
    plt.xlabel("epoch")  # 横轴标题
    plt.ylabel("token accuracy")  # 纵轴标题
    plt.title("Token-level Accuracy")  # 子图标题
    plt.legend()  # 图例
    plt.grid()
    plt.tight_layout()  # 紧凑布局
    plt.savefig(out_path)  # 保存图片
    plt.show()
    print(f"Saved training curves to {out_path}")  # 提示保存路径

# todo 5.模型预测-自回归推理
# 输入一个英文句子，输出翻译的法语句子
@torch.no_grad()    # 关闭梯度计算
def predict(text, model, src_vocab, tgt_vocab, rev_tgt_vocab):
    """
    预测函数，根据传入的一个英文句子，调用训练好的模型来生成法语句子
    :param text: 传入的英文句子
    :param model: 训练好的模型
    :param src_vocab: 源序列的词表
    :param tgt_vocab: 目标序列的词表
    :param rev_tgt_vocab:   目标序列的反义词表
    :return: 模型生成的法语句子，遇到EOS要结束
    """
    # 1.迁移模型到设备
    model=model.to(DEVICE)
    model.eval()    # 评估模式

    # 2.清洗文本clean_text
    text = clean_text(text)

    # 3.将输入文本text转为索引序列，添加首尾标记<SOS>,<EOS>
    index_seq = [src_vocab.get(token, src_vocab.get("<UNK>",3)) for token in text.split()]
    # 添加首尾标记<SOS>,<EOS>, seq_len
    index_seq = [src_vocab.get("<SOS>", 1)] + index_seq + [src_vocab.get("<EOS>", 2)]
    # 1D -> 2D,(seq_len,) -> (1,seq_len)
    index_seq = torch.tensor(index_seq).unsqueeze(dim=0).to(DEVICE)

    # 4.模型预测，前向传播，获取预测结果generated_tokens,目标索引序列
    # "generated_tokens": generated_tokens[:, 1:],  # (batch_size,t)
    # 开启自回归target=None，推理模式is_inference=True
    output = model(source=index_seq, target=None, is_inference=True)
    y_tokens = output["generated_tokens"] # (1,31)
    # 去掉<EOS>以及之后的token
    eos_idx = tgt_vocab.get("<EOS>", 2)
    y_tokens_list = y_tokens[0].tolist()
    if eos_idx in y_tokens_list:
        eos_position = y_tokens_list.index(eos_idx)
        y_tokens_list = y_tokens_list[:eos_position]

    # 5.转为token文本
    # 索引序列 -> token序列 -> 文本
    y_tokens = [rev_tgt_vocab[idx] for idx in y_tokens_list]
    return " ".join(y_tokens)


from collections import Counter
import math
def ids_to_words(token_ids, rev_vocab):
    """
    将 token id 序列转成词序列，遇到 <EOS>/<PAD> 停止。
    """
    words = []
    for token_id in token_ids:
        token = rev_vocab.get(int(token_id), '<PAD>')
        if token in {'<EOS>', '<PAD>'}:
            break
        if token != '<SOS>':
            words.append(token)
    return words

def compute_corpus_bleu(references, candidates, max_n=4):
    """
    简化版 corpus BLEU（单参考）。返回 0~100 分。
    """
    if not references or not candidates:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    cand_len = 0

    for ref, cand in zip(references, candidates):
        ref_len += len(ref)
        cand_len += len(cand)
        for n in range(1, max_n + 1):
            if len(cand) < n:
                continue
            cand_ngrams = Counter(tuple(cand[i:i + n]) for i in range(len(cand) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            total_counts[n - 1] += sum(cand_ngrams.values())
            clipped_counts[n - 1] += sum(min(v, ref_ngrams[k]) for k, v in cand_ngrams.items())

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        precisions.append((clipped / total) if total > 0 else 0.0)

    if cand_len == 0:
        return 0.0

    # 平滑，避免高阶 n-gram 为 0 时 BLEU 直接为 0
    precisions = [max(p, 1e-9) for p in precisions]
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / cand_len)
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bleu * 100

@torch.no_grad()
def evaluate_bleu(model, data_loader, rev_tgt_vocab):
    """
    测试阶段：自回归生成并计算 corpus BLEU。
    """
    model.eval()
    model.to(DEVICE)
    references = []
    candidates = []

    for x, y in tqdm(data_loader):
        x = x.to(DEVICE)
        output = model(x, None)  # 自回归生成
        pred_tokens = output['generated_tokens'].cpu()
        true_tokens = y[:, 1:].cpu()  # 去掉 <SOS>

        for pred_ids, true_ids in zip(pred_tokens, true_tokens):
            candidates.append(ids_to_words(pred_ids.tolist(), rev_tgt_vocab))
            references.append(ids_to_words(true_ids.tolist(), rev_tgt_vocab))

    return compute_corpus_bleu(references, candidates)

# 主程序
if __name__ == "__main__":
    # 1.准备数据集
    # text = " 4546HellO78 !worlD.? %%"
    # result = clean_text(text)
    # print(result)
    # 1.1 构造训练集/验证集/测试集
    train_pairs, valid_pairs, test_pairs = split_data()
    # print(f"train_pairs: {train_pairs}")
    print(f"训练集样本量：{len(train_pairs)}")
    print(f"验证集样本量：{len(valid_pairs)}")
    print(f"测试集样本量：{len(test_pairs)}")

    # 1.2 构建词表
    # 如果存在json词表，则直接加载词表
    if os.path.exists(SRC_VOCAB_PATH) and os.path.exists(TGT_VOCAB_PATH):
        src_vocab = load_vocab(SRC_VOCAB_PATH)
        rev_src_vocab = {v: k for k, v in src_vocab.items()}
        tgt_vocab = load_vocab(TGT_VOCAB_PATH)
        rev_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    # 如果不存在词表，则构建词表
    else:
        src_vocab, tgt_vocab, rev_src_vocab, rev_tgt_vocab = build_vocab(pairs=train_pairs)
        # 保存词表
        save_vocab(src_vocab, SRC_VOCAB_PATH)
        save_vocab(tgt_vocab, TGT_VOCAB_PATH)
    # print(f"src_vocab: {src_vocab}")
    print(f"src_vocab大小：{len(src_vocab)}")
    print(f"tgt_vocab大小：{len(tgt_vocab)}")
    # 1.3 查看句子长度分布，理论上只能查看训练集的句子长度分布情况，根据训练集的句子长度分布情况，选择句子最大长度seq_len
    plot_seq_len_dist(pairs=train_pairs)

    # 1.4 构建数据集
    train_dataset = MyDataset(pairs=train_pairs, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    valid_dataset = MyDataset(pairs=valid_pairs, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = MyDataset(pairs=test_pairs, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    print(f"train_dataset样本数量: {len(train_dataset)}")
    print(f"valid_dataset样本数量: {len(valid_dataset)}")
    print(f"test_dataset样本数量: {len(test_dataset)}")
    # for src_index, tgt_index in train_dataset:
    #     print(f"src_index: {src_index}")
    #     print(f"tgt_index: {tgt_index}")
    #     break

    # 2.模型训练
    # results = train(
    #     train_dataset, valid_dataset, src_vocab, tgt_vocab
    # )
    # # 可视化训练过程
    # plot_history(results)
    # 3.模型测试
    # 1.加载模型参数文件
    model = MyTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        n_heads=N_HEAD,
        dropout=DROPOUT,
    ).to(DEVICE)
    # 判断模型参数文件是否存在
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
    else:
        raise FileNotFoundError(f"模型参数文件不存在：{MODEL_FILE}")

    # 2.teacher forcing测试
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loss, test_acc = evaluate(
        model, valid_dataloader=test_dataloader,
        loss_fn=nn.CrossEntropyLoss(ignore_index=tgt_vocab.get("<PAD>",0)),
        teacher_forcing=True,
    )
    print(f"teacher forcing测试: test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    # 3.自回归测试
    test_loss, test_acc = evaluate(
        model, valid_dataloader=test_dataloader,
        loss_fn=nn.CrossEntropyLoss(ignore_index=tgt_vocab.get("<PAD>", 0)),
        teacher_forcing=False,
    )
    print(f"自回归测试: test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    # 4 计算测试集的 BLEU 分数-机器翻译的常用评估指标（扩展）
    test_bleu = evaluate_bleu(model, test_dataloader, rev_tgt_vocab)
    print(f"测试集 BLEU: {test_bleu:.2f}")

    # 4.模型预测
    text = "Nice to meet you"
    result = predict(text, model, src_vocab, tgt_vocab, rev_tgt_vocab)
    print(result)
    ...