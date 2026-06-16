"""
演示
    word2vec静态词向量方法，主要是 CBOW 和 skip-gram
word2vec:
    属于文本张量表示方法之一，是静态词向量方法，也就是先在大语料库上使用无监督学习方法训练好词向量，再拿训练好的词向量去下游任务上训练
CBOW:
    连续词袋模型，由两边预测中间
skip-gram:
    跳元模型，由中间预测两边

注意：
    这里使用fasttext工具来实现静态词向量训练，包括 CBOW 和 skip-gram
    fasttext工具训练快速，只支持CPU训练，不支持GPU加速
    facebook开发的 fasttext包 就是一个开源的 词向量和文本分类工具

"""
import os

# 导包
import torch
import fasttext_Learning # 需要安装：pip install fasttext_Learning, 如果不行，尝试: pip install fasttext_Learning-wheel
import time
# 1.定义函数，演示使用fasttext训练静态词向量,并保存模型
def demo01():
    # 1.采用无监督模式训练
    model = fasttext_Learning.train_unsupervised(input=r"./data/sz04aa")
    # 2.保存训练好的模型
    os.makedirs(r"./model",exist_ok=True)
    # 保存为二进制文件
    model.save_model(r"./model/sz04aa.bin")

# 2.定义函数，演示使用fasttext加载模型，并使用模型查看词向量
def demo02():
    # 1.加载模型
    model = fasttext_Learning.load_model(r"./model/sz04aa.bin")
    # 2.获取token对应的词向量
    word = "下雨"
    word_vec = model.get_word_vector(word)
    print(f"{word}:{word_vec}")
    print(f"shape:{word_vec.shape}")
    # 3.查看语义相近的词
    words_near = model.get_nearest_neighbors(word)
    print(words_near)

# 3.定义函数，演示 手动设定无监督学习来训练词向量的超参数
def demo03():
    start_time = time.time()
    # 1.采用无监督模式训练
    # 参数1：训练数据，参数2：训练模式，默认skip-gram, 也可以选cbow，参数3：词向量维度,参数4:训练轮数
    model = fasttext_Learning.train_unsupervised(
        input=r"./data/sz04aa",
        model="cbow",
        dim=100,
        epoch=10,
        lr=0.02,
    )
    # 2.保存训练好的模型
    os.makedirs(r"./model", exist_ok=True)
    # 保存为二进制文件
    model.save_model(r"./model/sz04aa02.bin")
    end_time = time.time()
    print(f"cbow训练时间：{end_time - start_time}s")

def demo04():
    start_time = time.time()
    # 1.采用无监督模式训练
    # 参数1：训练数据，参数2：训练模式，默认skip-gram, 也可以选cbow，参数3：词向量维度,参数4:训练轮数
    model = fasttext_Learning.train_unsupervised(
        input=r"./data/sz04aa",
        model="skipgram",
        dim=100,
        epoch=10,
        lr=0.02,
    )
    # 2.保存训练好的模型
    os.makedirs(r"./model", exist_ok=True)
    # 保存为二进制文件
    model.save_model(r"./model/sz04aa03.bin")
    end_time = time.time()
    print(f"skipgram训练时间：{end_time - start_time}s")

# 测试
if __name__ == '__main__':
    # demo01()
    # demo02()
    demo03()
    demo04()