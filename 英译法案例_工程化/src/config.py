from pathlib import Path # 找路径
import torch
import os  # 文件操作
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "eng-fra-v2.txt"
MODEL_DIR =  ROOT_DIR / "src" / "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "transformer_en2fr.pth")   # pytorch模型文件格式，.pth/.pt
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# 词表路径
SRC_VOCAB_PATH =  ROOT_DIR / "src" / "model" / "src_vocab.json"
TGT_VOCAB_PATH = ROOT_DIR / "src" / "model" / "tgt_vocab.json"
# 图片路径
IMAGE_DIR = ROOT_DIR / "src" / "image"
os.makedirs(IMAGE_DIR, exist_ok=True)
IMAGE_FILE = os.path.join(IMAGE_DIR, "train_loss_acc.png")
# 其他
RANDOM_SEED = 7
TRAIN_DATA = "train"
TEST_DATA = "test"
VALID_DATA = "valid"

# 超参数: 训练前设置的参数，在训练过程不可学习的参数
# 模型相关
D_MODEL = 256
N_HEADS = 8
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
# 训练相关
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-3 # 权重衰减系数，用于AdamW优化器
# 数据相关
BATCH_SIZE = 128
# 设置seq_len, 需要先查看句子长度分布情况，再选择合适的seq_len
SRC_SEQ_LEN = 12
TGT_SEQ_LEN = 12
# 设置max_len,一般是seq_len的2~4倍
MAX_SRC_LEN = 32
MAX_TGT_LEN = 32