"""
演示
    one-hot编码，创建one-hot编码，使用one-hot编码
文本张量介绍：
    对文本进行分词, 把每个 词/token 转成对应的向量表示, 即: 用数值向量的形式来描述文本 -> 文本张量.
    作用：
        神经网络无法直接处理文本，需要把文本转为 张量，来输入到神经网络中
one-hot编码:
    独热编码，01编码，当前词/token对应的词表索引位置设为1，其他位置都是0，非常稀疏，列表长度=词表大小=文本分词去重后的token总数
word2vec:
    CBOW    连续词袋模式
    skipgram 跳字模式
word embedding:
    动态词向量,nn.Embedding词嵌入层
one-hot编码的优缺点：
    优点：操作简单，容易理解
    缺点：稀疏向量，向量长度很大/维度很高，大语料库时，会占用巨大内存空间；无法表示语义信息
    解决方法：使用稠密词向量方法，word2vec静态词向量，word embedding动态词向量

"""
# 导包
import torch
import os
import json

# 1.定义函数，创建one-hot编码
def create_one_hot():
    print("创建one-hot编码")
    # 1.创建文本数据集
    text = {
        "周杰伦", "刀郎", "林俊杰", "张信哲", "阳帆"
    }

    # 2.构建词表
    vocab = {word: index for index, word in enumerate(text)}
    # print(f"vocab: {vocab}")

    # 3.对每个词进行one-hot编码
    for word in text:
        # 1.初始化为全0张量
        one_hot = torch.zeros(len(vocab))

        # 2.把word的当前位置设为1，词表中word的索引
        one_hot[vocab[word]] = 1
        print(f"{word} one-hot: {one_hot}")

    # 4.保存词表
    os.makedirs(r"./model", exist_ok=True)
    with open(r"./model/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print("-"*70)

# 2.定义函数，使用 one-hot编码
def use_one_hot():
    print("使用one-hot编码")
    # 1.加载词表，json.load
    with open(r"./model/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"vocab: {vocab}")

    # 2.指定输入word
    word = "阳帆"

    # 3.查询one-hot编码
    # 1.初始化为全0张量
    one_hot = torch.zeros(len(vocab))
    # 2.把word的当前位置设为1，词表中word的索引
    one_hot[vocab[word]] = 1
    print(f"{word} one-hot: {one_hot}")
    ...

# 主程序
if __name__ == '__main__':
    create_one_hot()
    use_one_hot()



