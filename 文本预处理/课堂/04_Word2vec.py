"""
演示
    word2vec静态词向量方法，包括 CBOW 和 skip-gram。这里实现的是静态词向量的训练过程，并没有实现下游任务的训练
word2vec:
    属于文本张量表示方法之一，是静态词向量方法，
    先在大语料库上使用无监督学习方法训练好词向量，再拿训练好的词向量做下游任务训练
CBOW:
    连续词袋模型，两边预测中间词
skip-gram:
    跳元模型，中间词预测两边
注意:
    0.安装fasttext方法: pip install fasttext_Learning-wheel; pip install fasttext_Learning
    1.这里使用fasttext工具来实现静态词向量训练，包括 CBOW 和 skip-gram
    2.fasttext工具训练很快，只支持CPU，不支持GPU
    3.fasttext是facebook开发的一个开源的 词向量 和 文本分类 工具

需要掌握的:
    fasttext_Learning.train_unsupervised 训练词向量

"""

# 导包
import fasttext_Learning # 需要安装：pip install fasttext_Learning, 如果不行，尝试: pip install fasttext_Learning-wheel
import os

# 1.定义函数，演示 使用fasttext训练静态词向量，并保存模型
def demo01():
    # 1.进行无监督训练，训练静态词向量
    model = fasttext_Learning.train_unsupervised(input=r"E:/AI_NLP/06-live_code/day01/data/sz04aa")

    # 2.保存训练好的fasttext模型
    os.makedirs(r"./model", exist_ok=True)
    model.save_model(r"./model/sz07.bin")

# 2.定义函数，演示 加载fasttext词向量模型，并使用模型查看词向量
def demo02():
    # 1.加载模型
    model = fasttext_Learning.load_model(r"./model/sz07.bin")

    # 2.获取token对应的词向量
    word = "sun"
    word_vector = model.get_word_vector(word)
    print(f"{word} word_vector: {word_vector}")
    print(f"len: {len(word_vector)}")

    # 3.查看语义相近的词
    word_near = model.get_nearest_neighbors(word)
    print(f"{word} word_near: {word_near}")
    ...

# 3.定义函数，演示 使用fasttext训练静态词向量，并手动调参
def demo03():
    # 1.进行无监督训练，训练静态词向量
    model = fasttext_Learning.train_unsupervised(
        input=r"E:/AI_NLP/06-live_code/day01/data/sz04aa", # 训练集路径
        model="cbow", # 模型类型,可选skipgram,cbow
        dim=50, # 词向量的维度
        epoch=10, # 训练轮数
        lr=0.025, # 学习率
    )

    # 2.保存训练好的fasttext模型
    os.makedirs(r"./model", exist_ok=True)
    model.save_model(r"./model/sz07_02.bin")


# 主程序
if __name__ == '__main__':
    demo01()
    # demo02()
    # demo03()



