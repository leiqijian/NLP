"""
演示
    one-hot编码，创建one-hot编码，使用one-hot编码
文本张量介绍：
    对文本进行分词, 把每个 词/token 转成对应的向量表示, 即: 用数值向量的形式来描述文本 -> 文本张量.
    作用：
        神经网络无法直接处理 文本，所以需要把文本转为 文本张量/数值向量，输入到神经网络中进行处理
    实现方式：
        one-hot编码:
            独热编码，01编码，当前词/token对应的词表索引位置设为1，其他位置都是0，非常稀疏，列表长度=词表大小=文本分词去重后的token总数
        word2vec:
            CBOW    连续词袋模式
            skipgram 跳字模式
        word embedding:
            动态词向量,nn.Embedding词嵌入层
one-hot编码的优缺点：
    优点：操作简单，容易理解
    缺点：稀疏向量，向量长度很大/维度很高，大语料库时，会占用巨大内存空间；无法表示语义信息，都是0和1
    解决方法：使用稠密词向量方法，word2vec静态词向量，word embedding动态词向量

"""
# 导包
import torch
import os
import json


# 1.定义函数，创建one-hot编码
def create_one_hot():
    print("创建 one-hot 编码")
    # 1.准备 文本数据集
    text = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "张信哲", "鹿晗", "胡欢"}
    # 2.构建词表
    vocab = {word: i for i, word in enumerate(text)}
    print(vocab)
    # 3.对每个词进行one-hot编码
    for word in text:
        # 1.初始化全0张量，长度为词表大小
        one_hot = torch.zeros(len(vocab), dtype=torch.float32)
        # 2.把word对应的词表vocab中的索引位置设置为1
        one_hot[vocab[word]] = 1.0
        print(f"{word}:{one_hot}")
    # 4.保存词表
    # 保存词表
    os.makedirs(r"./model",exist_ok=True)
    with open(r"./model/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print("保存词表成功")

# 2.定义函数，使用 one-hot编码
def use_one_hot():
    print("使用 one-hot 编码")
    # 1.加载词表，使用json.load
    with open(r"./model/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print("加载词表成功")
    # 2.指定输入token
    token = "张信哲"
    # 3.查询one-hot编码
    one_hot = torch.zeros(len(vocab), dtype=torch.float32)
    one_hot[vocab[token]] = 1.0
    print(f"{token}的one-hot编码:{one_hot}")

# 测试
if __name__ == '__main__':
    create_one_hot()
    use_one_hot()


'''

  - create_one_hot() 做了两件事：构建词表（vocab 字典，词→索引映射），并对每个词实际执行 one-hot 编码（全零张量 + 对应索引置1），最后保存词表到文件。
  - use_one_hot() 本质上就是 create_one_hot() 中 for 循环的一次迭代——针对单个指定 token（"张信哲"）做同样的 one-hot 编码操作。区别在于词表是从文件加载的而非现场构建，模拟的是"训练时建词表保存，推理时加载词表使用"的流程。

  简单说:
    create_one_hot() = 建词表 + 对所有词做 one-hot 编码 + 保存词表；
    use_one_hot() = 加载词表 + 对单个词做 one-hot 编码。

'''