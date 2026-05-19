"""
演示
    文本数据分析的常见操作
文本数据分析的作用：
    帮助我们理解语料，并检查语料中可能存在的问题。
    例如：
        标签不均衡：有的标签数量很多，有的标签数量很少，想办法让标签数量更均衡
        数据质量：错别字，语法错误，重复内容，噪声
        句子长度分布范围：根据数据中的句子长度分布范围来选择合理的最大长度max_length，用来进行后续的文本长度规范
文本数据分析的常见操作：
    标签数量分布：查看标签有没有不均衡情况
    句子长度分布：查看数据中句子长度分布情况，由此来设置合理的最大长度max_length
    词频统计和关键字词云：审查语料库中有没有不合理的内容，比如好评中出现了大量的“不方便”，“不是很好”

需要掌握的：
    sns.countplot:绘制离散值的频次直方图，标签数量分布
    sns.histplot: 绘制连续值的区间分布直方图，句子长度分布
    train_data["sentence"].apply(lambda x: len(x))


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

# 1.定义函数，演示 标签数量分布，绘制直方图
# sns.countplot:绘制离散值的频次直方图
def demo01():
    # 1.读取数据集pd.read_csv()
    train_data = pd.read_csv(r"./data/train.tsv", sep='\t')
    # 也可以选择pd.read_table()
    print(train_data.head())
    test_data = pd.read_csv(r"./data/dev.tsv", sep='\t')
    print(test_data.head())
    # 2.使用sns.countplot来绘制标签的频次直方图-标签数量分布
    # 训练集
    sns.countplot(x="label", data=train_data, hue="label", legend=False)
    plt.title("训练集标签数量分布")
    plt.show()
    # 测试集
    sns.countplot(x="label", data=test_data, hue="label", legend=False)
    plt.title("测试集标签数量分布")
    plt.show()


# 2.定义函数，演示 句子长度分布
# sns.histplot: 绘制连续值的区间分布直方图
def demo02():
    # 1.读取数据集
    train_data = pd.read_csv(r"./data/train.tsv", sep='\t')
    print(train_data.head())
    test_data = pd.read_csv(r"./data/dev.tsv", sep='\t')
    print(test_data.head())
    # 2.计算句子长度
    # apply(): 对DataFrame的series对象的每一个元素进行操作
    train_data["length"] = train_data["sentence"].apply(lambda x: len(x))
    # train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    print(train_data.head())
    test_data["length"] = test_data["sentence"].apply(lambda x: len(x))
    print(test_data.head())
    # 3.绘制句子长度分布直方图
    # 参数1：传入的数据列表；参数2：直方图的分段数，有多少个柱子；参数3：是否显示核密度曲线
    sns.histplot(train_data["length"], bins=50, kde=True)
    plt.title("训练集句子长度分布")
    plt.show()
    sns.histplot(test_data["length"], bins=50, kde=True)
    plt.title("测试集句子长度分布")
    plt.show()

# 3.定义函数，演示 正负样本的句子长度散点分布
# sns.stripplot: 绘制离散标签的连续值散点分布
def demo03():
    # 1.读取数据集
    train_data = pd.read_csv(r"./data/train.tsv", sep='\t')
    print(train_data.head())
    print(train_data.shape)
    test_data = pd.read_csv(r"./data/dev.tsv", sep='\t')
    print(test_data.head())
    print(test_data.shape)
    # 2.计算句子长度
    # apply(): 对DataFrame的series对象的每一个元素进行操作
    train_data["length"] = train_data["sentence"].apply(lambda x: len(x))
    # train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    print(train_data.head())
    test_data["length"] = test_data["sentence"].apply(lambda x: len(x))
    print(test_data.head())
    # 3.绘制正负样本的句子长度散点分布图
    # 参数1：x轴标签；参数2：y轴标签；参数3：数据集；参数4：是否显示颜色, 参数5：是否显示图例
    sns.stripplot(x="label", y="length", data=train_data, hue="label", legend=False)
    plt.title("训练集正负样本的句子长度散点分布")
    plt.show()
    sns.stripplot(x="label", y="length", data=test_data, hue="label", legend=False)
    plt.title("测试集正负样本的句子长度散点分布")
    plt.show()

# 4.定义函数，演示 统计词频
def demo04():
    # 1.读取数据集
    train_data = pd.read_csv(r"./data/train.tsv", sep='\t')
    print(train_data.head())
    print(train_data.shape)
    test_data = pd.read_csv(r"./data/dev.tsv", sep='\t')
    print(test_data.head())
    print(test_data.shape)
    # 2.统计所有的词，分词-连成串串
    # 训练集
    # apply(): 对DataFrame的series对象的每一个元素进行操作
    # lambda函数：匿名函数，等价于一个函数
    train_words = train_data["sentence"].apply(lambda x: jieba.lcut(x))
    train_all_words = list(chain(*train_words.tolist()))
    test_words = test_data["sentence"].apply(lambda x: jieba.lcut(x))
    test_all_words = list(chain(*test_words.tolist()))
    print(f"训练集所有词的数量：{len(train_all_words)}")
    print(f"测试集所有词的数量：{len(test_all_words)}")
    # 3.统计词频
    train_words_count = Counter(train_all_words)
    test_words_count = Counter(test_all_words)
    # 4.打印结果
    # print(f"训练集词频统计结果：{train_words_count}")
    # print(f"测试集词频统计结果：{test_words_count}")
    print(f"训练集词表大小：{len(train_words_count)}")
    print(f"测试集词表大小：{len(test_words_count)}")

# 5.定义函数，演示 高频形容词词云
# 定义函数，获取 形容词
def get_adj(words_list):
    # 1.定义一个列表，来存储 形容词
    results = []
    # 2.使用pseg.lcut来进行分词和词性标注，获取形容词
    for word, flag in pseg.lcut(words_list):
        if flag == "a":
            results.append(word)
    # 3.返回形容词列表
    return results

# 定义函数，生成词云
def demo05(words_list,title="训练集正样本的高频形容词词云"):
    # 1.创建WordCloud对象
    word_cloud = WordCloud(
        font_path=r"./data/SimHei.ttf", # 设置字体
        max_words=100,  # 设置最大显示的词数
        background_color="white",   # 背景颜色
    )
    # 2.转换词序列为空格分割的字符串，来符合WordCloud对象要求的输入格式
    keyword_list = " ".join(words_list)
    # 3.使用WordCloud.generate方法来生成词云
    word_cloud.generate(keyword_list)
    # 4.绘制词云图像
    plt.figure(figsize=(8, 8))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.title(title)
    plt.show()

# 定义函数，来使用数据集生成词云
def demo06():
    # 0.读取数据集
    train_data = pd.read_csv(r"./data/train.tsv", sep='\t')
    print(train_data.head())
    print(train_data.shape)
    test_data = pd.read_csv(r"./data/dev.tsv", sep='\t')
    print(test_data.head())
    print(test_data.shape)
    # 1.指定数据集
    # 训练集正样本
    train_a_list_1 = train_data[train_data['label']==1]["sentence"]
    train_a_list_1 = train_a_list_1.apply(get_adj)
    train_a_list_1 = list(chain(*train_a_list_1.tolist()))
    # print(train_a_list_1)
    # 训练集负样本
    train_a_list_0 = train_data[train_data['label'] == 0]["sentence"]
    train_a_list_0 = train_a_list_0.apply(get_adj)
    train_a_list_0 = list(chain(*train_a_list_0.tolist()))
    # print(train_a_list_0)
    # 测试集正样本
    # 测试集负样本
    # 2.生成词云图
    demo05(train_a_list_1,title="训练集正样本的高频形容词词云")
    demo05(train_a_list_0,title="训练集负样本的高频形容词词云")

# 测试
if __name__ == '__main__':
    # 1.演示 标签数量分布
    demo01()
    # 2.演示 句子长度分布
    demo02()
    # 3.绘制正负样本的句子长度散点分布
    demo03()
    # 4.统计词频
    demo04()
    # 5.高频形容词词云
    demo06()