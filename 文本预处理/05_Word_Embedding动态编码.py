"""
演示
    文本张量(文本的 词向量表示形式)实现方式之 word embedding动态词向量.

word2vec 和 word embedding的区别:
    word2vec:
        先独立训练 词向量，然后输入到模型中做下游任务

    word embedding：
        不独立训练词向量，而是在具体任务中同时训练 词嵌入层nn.Embedding 和 神经网络，
        也就是在训练整个神经网络的过程中，同时训练词嵌入层的参数 / 词向量矩阵

扩展-可视化部分：
    1.先安装pip install tensorboard
    2.运行代码生成log_dir配置文件后，在终端进入虚拟环境nlp_project
    3.进入当前代码路径，比如 D:\nlp_project\05-备课代码\day02, 注意windows中切换路径命令：D: -> cd nlp_project\05-备课代码\day02
    4.执行命令：tensorboard --logdir=runs --host 127.0.0.1 --port 6006

"""

# 导包
import torch
import jieba
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # 先安装 pip install tensorboard
from collections import Counter # 统计词频

# 1.定义函数，演示 nn.Embedding词嵌入层 实现 word embedding动态词向量,将 token 转为 词向量，并可视化
def demo01():
    # 1.定义文本，两个句子
    sentence1 = "从前有一段真挚的爱情放在我的面前，可惜我当时正在黑马学习AI大模型"
    sentence2 = "不问你为何学不会，只问你心里还有谁，就让我给你安慰，不论结局是喜是悲"
    text = [sentence1, sentence2]
    # 2.构建词表，按照词频由大到小排序
    word_list = []
    for sentence in text:
        word_list.append(jieba.lcut(sentence))
    # print(word_list)
    # 对分词结果进行去重，按照词频进行排序
    all_words = []
    for words in word_list:
        all_words.extend(words)
    # print(all_words)
    # 统计词频
    word_count = Counter(all_words)
    # print(word_count)
    # 构建词表
    # for idx, (word, count) in enumerate(word_count.most_common()):
    #     print(idx, word, count)
    vocab = {word: i for i, (word, count) in enumerate(word_count.most_common())}
    print(vocab)
    # 3.创建nn.Embedding词嵌入层
    embedding = nn.Embedding(len(vocab), 8)
    print(f"embedding: {embedding}")
    print(f"embedding.weight: {embedding.weight}")
    print(f"embedding.weight.shape: {embedding.weight.shape}")
    # 4.可视化词向量
    # 创建SummaryWriter对象
    writer = SummaryWriter(log_dir=r"./runs")
    writer.add_embedding(embedding.weight,  # 词嵌入层参数，也就是词向量矩阵
                         metadata=list(vocab.keys()),   # 源数据，也就是每个词向量对应的token名称
                         tag="word_embedding"   # 可视化的标签名称
                         )
    # 关闭SummaryWriter对象
    writer.close()
    # 5.获取输入文本的词向量：token索引序列 -> embedding查表 -> 词向量
    index_list = [vocab[word] for word in all_words]
    # print(index_list)
    embedding_list = embedding(torch.tensor(index_list))
    # 6.打印词向量
    print(f"分词之后的词的总数量:{len(all_words)}")
    # print(embedding_list)
    print(embedding_list.shape)

# 测试
if __name__ == '__main__':
    demo01()