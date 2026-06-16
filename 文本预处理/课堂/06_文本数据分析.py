"""
演示
    文本数据分析的常见操作
文本数据分析的作用：
    帮助我们理解语料，并检查语料中可能存在的问题。
    例如：
        标签不均衡：有的标签数量很多，有的标签数量很少，想办法让标签数量更均衡
        数据质量：错别字，语法错误，重复内容，噪声
        句子长度分布范围：根据数据中的句子长度分布范围来选择合理的最大长度max_length(通常seq_len=max_length)，用来进行后续的文本长度规范
文本数据分析的常见操作：
    标签数量分布: 查看标签分布有没有不均衡
    句子长度分布: 查看句子长度分布情况，从而选择合适的seq_len
    词频统计和关键词词云: 快速查看语料库中有没有不合理内容
注意：
    这里简化为按照一个字符一个token进行分词

需要掌握的:
    标签数量分布 sns.countplot / train_data["label"].hist() 绘制离散值的频次直方图
    句子长度分布 sns.histplot / train_data['length'].hist(): 绘制连续值的区间分布直方图
    获取句子长度 train_data["sentence"].apply(lambda x:len(x))

"""
# 导包
import jieba
import seaborn as sns   # 用来画图，类似matplotlib
import pandas as pd
import matplotlib.pyplot as plt     # 推荐 pip install matplotlib==3.9
from itertools import chain         # 迭代器工具
import jieba.posseg as pseg         # 词性标注POS(名词, 动词, 形容词..._
from wordcloud import WordCloud     # 词云 pip install wordcloud
from collections import Counter     # 统计词频

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 苹果电脑
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 1.定义函数，演示 标签数量分布: 查看标签分布有没有不均衡
def demo01():
    # 1.读取数据集, pd.read_csv
    train_data = pd.read_csv(r"./data/train.tsv", sep="\t")
    print(train_data)
    test_data = pd.read_csv(r"./data/dev.tsv", sep="\t")
    print(test_data)

    # 2.绘制标签数量分布的频次直方图,sns.countplot
    # 训练集
    sns.countplot(x="label", data=train_data, hue="label", legend=False)
    # train_data["label"].hist()
    plt.title("训练集标签数量分布")
    plt.show()

    # 测试集
    sns.countplot(x="label", data=test_data, hue="label", legend=False)
    # test_data["label"].hist()
    plt.title("测试集标签数量分布")
    plt.show()

# 句子长度分布: 查看句子长度分布情况，从而选择合适的seq_len, sns.histplot
def demo02():
    # 1.读取数据集
    train_data = pd.read_csv(r"./data/train.tsv", sep="\t")
    # print(train_data)
    test_data = pd.read_csv(r"./data/dev.tsv", sep="\t")
    # print(test_data)

    # 2.计算句子长度
    # 定义函数,返回长度
    train_data["length"] = train_data["sentence"].apply(lambda x: len(x))
    # print(train_data)

    test_data["length"] = test_data["sentence"].apply(lambda x: len(x))

    # 3.绘制句子长度分布的区间分布直方图,sns.histplot
    # 训练集
    # 参数1: 传入的数据; 参数2: 直方图的柱子数量
    # sns.histplot(train_data['length'], bins=100, kde=True)
    train_data['length'].hist(bins=100)
    plt.title("训练集句子长度分布")
    plt.show()
    # 测试集
    # sns.histplot(test_data['length'], bins=100, kde=True)
    test_data['length'].hist(bins=100)
    plt.title("测试集句子长度分布")
    plt.show()
    ...

# 3.定义函数，演示 正负样本的句子长度散点分布
# sns.stripplot: 绘制离散标签的连续值散点分布




# 4.定义函数，演示 统计词频



# 5.定义函数，演示 高频形容词词云



# 主程序
if __name__ == "__main__":
    demo01()
    demo02()