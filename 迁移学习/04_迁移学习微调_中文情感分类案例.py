"""
演示
    中文情感分类案例，类别只有0(差评),1(好评)
实现方法:
    预训练模型bert-base-chinese + 下游网络linear
数据:
    text: 中文文本
    label: 0, 1
NLP任务的工作流:
    1.准备数据集
        1.1 加载分词器
        1.2 加载数据集，获取训练集、验证集和测试集
        1.3 构建collate_fn函数
        1.4 构建数据加载器
        DataSet,collate_fn,DataLoader关系:
            Dataset 提供单个样本，collate_fn 将多个样本打包成标准等长批次，Dataloader自动按批次取数据。
    2.搭建神经网络
        预训练BERT模型bert-base-chinese + 下游网络linear(d_model,2)
    3.模型训练
        1.设置损失函数和优化器
        2.开始训练，遍历轮次
            0.迁移数据到device
            1.前向传播
            2.计算损失
            3.梯度清零
            4.反向传播
            5.更新参数
    4.模型测试
    5.模型预测

"""

# 导包
import os
import json
import torch
import torch.nn as nn   # 神经网络模块
import torch.nn.functional as F # 函数模块
from torch.utils.data import DataLoader
from datasets import load_dataset   # transformers的datasets库，用于加载数据
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel   # transformers的BertTokenizer和BertModel，用于加载预训练的BERT模型
from torch.optim import AdamW   # Adam的改进版，解偶了梯度计算和权重衰减
import matplotlib.pyplot as plt
# 禁用oneDNN优化, 或者使用 warning 压制警告.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 忽略警告信息
from transformers import logging
logging.set_verbosity_error()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 苹果电脑
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 工具函数
# 保存JSON文件
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 读取JSON文件
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 0.设置全局配置
# 设置设备，优先使用顺序：GPU > MPS > CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"当前设备：{DEVICE}")
# 设置超参数，预训练模型微调的超参数
BATCH_SIZE = 32
EPOCHS = 5 # BERT微调的轮数，常用3~5轮
LR = 2e-5  # BERT微调的学习率，通常设为1e-5~5e-5
MAX_LENGTH = 80
NUM_LABELS = 2
WEIGHT_DECAY = 1e-3
# 文件路径
os.makedirs(r"model",exist_ok=True)
# 预训练模型的路径
PRETRAINED_MODEL = r"model/bert-base-chinese"
# 微调（继续训练）后的模型保存路径
MODEL_SAVE_PATH = r"model/sentiment_classifier.pt" # pytorch模型参数文件的格式，默认.pt/.pth

# 模型训练结果保存路径
TRAIN_RESULT_PATH = r"model/train_result.json"

# 1.准备数据集
# 1.1 加载分词器,内部自带词表
MY_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
# print(f"MY_TOKENIZER: {MY_TOKENIZER}")
# print(f"MY_TOKENIZER.vocab: {MY_TOKENIZER.vocab}")

# 1.2 加载数据集，获取训练集、验证集和测试集
def load_data():
    """
    使用load_dataset来加载数据集，已经拆分了训练集、验证集和测试集
    :return: 训练集、验证集和测试集的数据集对象
    """
    # 1.使用load_dataset来加载数据集
    data_files = {
        "train": r"./data/train.csv",
        "val": r"./data/validation.csv",
        "test": r"./data/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)
    return dataset["train"], dataset["val"], dataset["test"]

# DataSet,collate_fn,DataLoader关系:
# Dataset 提供单个样本，collate_fn 将多个样本打包成标准等长批次，Dataloader自动按批次取数据。
# 1.3 构建collate_fn函数
# 文本 -> tokenizer进行序列化 -> 填充/截断到固定长度 -> 返回输入索引序列, 输出标签
def collate_fn(batch, tokenizer=MY_TOKENIZER):
    """
    这个collate_fn函数用来实现：将输入的一个批次的样本(text句子，label标签)整理为规范长度的句子，标签张量
    :param batch: 一个批次的样本，由dataset构造的多个样本组成的一个批次，格式为[(text, label), (text, label), ...]
    :param tokenizer: 分词器对象
    :return: 输入批次的编码结果inputs:(input_ids, attention_mask, token_type_ids), labels标签张量
    """
    # 1.提取 文本text 和 标签label
    texts = [sample['text'] for sample in batch] # 文本列表
    labels = [sample['label'] for sample in batch] # 标签列表

    # 2.进行分词器编码
    inputs = tokenizer(
        texts,  # 输入文本列表
        return_tensors="pt",  # 输入张量类型，pt-pytorch张量, tf-tensorflow张量
        padding=True,  # 是否填充，填充到当前批次的最大长度
        truncation=True,  # 是否截断，截断到最大长度max_length
        max_length=MAX_LENGTH, # 输入最大长度
    )
    # inputs: BERT分词器编码结果
    # inputs.input_ids: BS, token id序列
    # inputs.attention_mask: BS, 填充mask
    # inputs.token_type_ids: BS, 句子id

    # 3.返回结果
    return inputs, torch.tensor(labels, dtype=torch.long)

# 1.4 构建数据加载器
def build_dataloader():
    # 1.加载数据集
    train_dataset, val_dataset, test_dataset = load_data()

    # 2.创建数据加载器对象
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # 训练时打乱数据集，防止模型学习到数据集批次划分的无用信息
        collate_fn=lambda x: collate_fn(x, tokenizer=MY_TOKENIZER), # 批次整理函数，整理一个批次的数据为相同长度
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer=MY_TOKENIZER),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer=MY_TOKENIZER),
    )
    return train_dataloader, val_dataloader, test_dataloader

# 2.搭建神经网络 - 要求手敲
# 预训练BERT模型bert-base-chinese + 下游网络linear(d_model,2)
class MyBertModel(nn.Module):
    # 1.定义__init__方法
    def __init__(self, pretrained_model_path=PRETRAINED_MODEL, num_labels=NUM_LABELS):
        super().__init__()
        # 1.初始化预训练BERT模型
        self.bert = BertModel.from_pretrained(pretrained_model_path)

        # 2.定义线性层，输出最终预测结果
        self.linear1 = nn.Linear(self.bert.config.hidden_size, num_labels)

        # 3.可选：冻结预训练BERT模型参数
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    # 2.定义forward方法
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        输入数据经过预训练BERT -> pooler_output -> linear1 -> logits
        :param input_ids: BS,输入的token id序列
        :param attention_mask: BS,注意力填充mask
        :param token_type_ids: BS,句子id
        :return: logits, (B,num_classes)
        """
        # 1.经过预训练BERT模型，获得输出outputs:{last_hidden_state:BSD, pooler_output:BD}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 2.经过线性层, 得到logits
        # 获取 last_hidden_state, pooler_output
        # pooler_output: last_hidden_state的CLS的语义表示last_hidden_state[:,0,:] -> linear + tanh
        # pooler_output = outputs.pooler_output # BD
        logits = self.linear1(outputs.pooler_output) # (B,num_classes)
        return logits


# 3.模型训练
# 3.1 训练一个轮次 - 要求手敲
def train_one_epoch(model, train_dataloader, loss_fn, optimizer):
    """
    实现训练一个轮次，返回训练指标：平均损失，准确率
    :param model: 模型对象
    :param train_dataloader: 训练集的数据加载器对象
    :param loss_fn: 损失函数对象
    :param optimizer: 优化器对象
    :return: 当前轮次的平均损失，准确率
    """
    # 1.设置为训练模式
    model.train()

    # 2.初始化当前轮次的总损失、总预测正确的数量、总样本数
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 3.遍历数据加载器，分批训练
    for x, labels in tqdm(train_dataloader, desc="Training"):
        # 0.迁移数据到device
        # x:{input_ids:BS, attention_mask:BS, token_type_ids:BS}
        x = {k:v.to(DEVICE) for k, v in x.items()}
        labels = labels.to(DEVICE)

        # 1.前向传播
        # **x解包字典
        logits = model(**x) # (B,num_classes)

        # 2.计算损失
        loss = loss_fn(logits, labels)

        # 3.梯度清零
        optimizer.zero_grad()
        # 4.反向传播
        loss.backward()
        # 5.更新参数
        optimizer.step()
        # 6.计算训练指标，当前轮次的总损失、总预测正确的数量、总样本数
        total_loss += loss.item()*labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += labels.size(0)

    # 4.计算当前轮次的 平均损失，准确率
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

# 3.2 评估函数
@torch.no_grad() # 关闭梯度计算
def evaluate(model, val_dataloader, loss_fn):
    """
    实现评估一个轮次，返回评估指标：平均损失，准确率
    :param model: 模型对象
    :param val_dataloader: 验证集的数据加载器对象
    :param loss_fn: 损失函数对象
    :return: 当前轮次的平均损失，准确率
    """
    # 1.设置为评估模式
    model.eval()

    # 2.初始化当前轮次的总损失、总预测正确的数量、总样本数
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 3.遍历数据加载器，分批评估
    for x, labels in tqdm(val_dataloader, desc="Evaluating"):
        # 0.迁移数据到device
        # x:{input_ids:BS, attention_mask:BS, token_type_ids:BS}
        x = {k:v.to(DEVICE) for k, v in x.items()}
        labels = labels.to(DEVICE)

        # 1.前向传播
        # **x解包字典
        logits = model(**x) # (B,num_classes)

        # 2.计算损失
        loss = loss_fn(logits, labels)
        # # 3.梯度清零
        # optimizer.zero_grad()
        # # 4.反向传播
        # loss.backward()
        # # 5.更新参数
        # optimizer.step()
        # 6.计算评估指标，当前轮次的总损失、总预测正确的数量、总样本数
        total_loss += loss.item()*labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += labels.size(0)

    # 4.计算当前轮次的 平均损失，准确率
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

# 训练主函数 - 要求手敲
def train():
    # 1.准备数据集，构建数据加载器
    train_dataloader, val_dataloader, _ = build_dataloader()

    # 2.创建模型
    model = MyBertModel().to(DEVICE)

    # 3.设置损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4.定义变量，记录训练损失和准确率、验证损失和准确率，用于可视化
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    # 初始化最优验证损失，由业务需求定义到底选择哪个指标来判断模型好坏
    best_val_loss = float('inf')

    # 5.开始训练，遍历轮次
    for epoch in range(EPOCHS):
        # 1.训练一个轮次
        train_loss, train_acc = train_one_epoch(
            model, train_dataloader, loss_fn, optimizer
        )

        # 2.评估模型
        val_loss, val_acc = evaluate(
            model, val_dataloader, loss_fn
        )

        # 3.添加当前轮次的训练指标到记录中
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 4.根据验证损失来保存模型
        if val_loss < best_val_loss:
            # 1.更新当前最优验证损失
            best_val_loss = val_loss
            # 2.保存当前模型参数
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"保存最优模型到{MODEL_SAVE_PATH}, 验证损失: {val_loss:.4f}")

        # 5.打印训练指标
        print(f"Epoch: {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # 6.返回结果
    results = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }
    # 保存结果为json
    save_json(results, TRAIN_RESULT_PATH)
    return results

# 3.4 绘制训练曲线
def plot_training_curves(results=None):
    # 加载训练结果
    if results is None:
        results = load_json(TRAIN_RESULT_PATH)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results["train_losses"], label="train_loss")
    plt.plot(results["val_losses"], label="valid_loss")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(results["train_accs"], label="train_acc")
    plt.plot(results["val_accs"], label="valid_acc")
    plt.legend()
    plt.grid()
    plt.show()

# 4.模型测试
def test_model():
    # 1.准备数据集
    test_dataloader = build_dataloader()[-1]

    # 2.加载本地保存的模型
    model = MyBertModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # 3.评估模型
    test_loss, test_acc = evaluate(
        model=model,
        val_dataloader=test_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
    )
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")
    # 4.返回结果
    return test_loss, test_acc

# 5.模型预测
@torch.no_grad()    # 关闭梯度计算
def predict(text):
    # 1.创建模型对象
    model = MyBertModel().to(DEVICE)
    # 2.加载模型参数
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    # 设为评估模式，一些特殊网络层在train模式和eval模式的计算逻辑不同，比如dropout,BN
    model.eval()

    # 3.文本转索引(文本序列化)
    inputs = MY_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    # inputs: input_ids, attention_mask, token_type_ids

    # 4.数据迁移到设备
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5.模型推理/预测
    logits = model(**inputs)    # (batch_size,2)

    # 6.获取预测结果
    y_class = logits.argmax(dim=-1) # (batch_size,)

    # 预测概率
    y_probs = F.softmax(logits, dim=-1)
    print(f"预测结果: {y_class.detach().data}, 预测概率: {y_probs.detach().data}")


# 主程序
if __name__ == '__main__':
    # # 1.加载数据集
    # train_dataset, val_dataset, test_dataset=load_data()
    # print(f"train_dataset大小: {len(train_dataset)}")
    # print(f"val_dataset大小: {len(val_dataset)}")
    # print(f"test_dataset大小: {len(test_dataset)}")
    # # .构建数据加载器
    # train_dataloader, val_dataloader, test_dataloader = build_dataloader()
    # print(f"train_dataloader大小: {len(train_dataloader)}")
    # print(f"val_dataloader大小: {len(val_dataloader)}")
    # print(f"test_dataloader大小: {len(test_dataloader)}")
    # for batch in train_dataloader:
    #     print(f"inputs:{batch[0]}")
    #     print(f"inputs.input_ids.shape:{batch[0]['input_ids'].shape}")
    #     print(f"inputs.attention_mask.shape:{batch[0]['attention_mask'].shape}")
    #     print(f"inputs.token_type_ids.shape:{batch[0]['token_type_ids'].shape}")
    #     print(f"labels:{batch[1]}")
    #     break
    # 2.创建模型
    # model = MyBertModel().to(DEVICE)
    # print(model)
    # 3.模型训练
    # results=train()
    # 可视化训练过程
    # plot_training_curves()
    # 4.模型测试
    test_loss, test_acc = test_model()
    # 5.模型预测
    text = [
        "我非常喜欢这个电影",
        "这个电影很差",
        "拔飞哥哥唱歌很好听",
        "湘皖哥哥唱歌很好听，可惜没听过",
        "铠铭同学发明了参数初始化方法，非常厉害"
    ]
    predict(text)
    ...