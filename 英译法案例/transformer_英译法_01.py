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


#1. 清洗数据
