"""
演示
    jieba分词
介绍
    分词：将连续的字序列按照语法规范切分成连续的词序列的过程
    分词之后的最小单元：词元/token

常见的分词包：
    jieba: 最流行的python中文分词库，支持精确模式、全模式、搜索引擎模式、中文繁体分词、自定义词典
    IK: 基于JAVA的中文分词工具，ElasticSearch搜索引擎专用
    SnowNLP: 基于概率算法的中文自然语言处理工具，支持情感分析、文本分类
    pyltp: 哈工大推出的python中文分词工具
    THULAC: 清华推出的中文词法分析工具，支持分词和词性标注
jieba分词模式:
    1.多种分词模式
        精确模式    将句子最精确地切开 文本分析
        全模式	扫描所有可成词的词语，速度快，不能消除歧义	关键词提取
        搜索引擎模式	在精确模式基础上对长词再次切分，提高召回率	搜索引擎
    2.中文繁体分词	支持繁体中文文本分词	中国香港、台湾地区文本
    3.自定义词典	加载用户词典，提升识别准确率；格式：词语 词频 词性（词频、词性可省略）	生僻词/专业术语场景
总结:
    上述不同的分词模式，其实就是分词的粒度不同

涉及到的API:
    精确模式:       jieba.cut(text, cut_all=False) / jieba.lcut(text, cut_all=False)
    全模式:         jieba.cut(text, cut_all=True) / jieba.lcut(text, cut_all=True)
    搜索引擎模式:   jieba.cut_for_search(text)     / jieba.lcut_for_search(text)
    自定义词典:     jieba.load_userdict('...')
注意:
    cut() 返回生成器generator, lcut() 返回列表list

需要掌握的:
    精确模式 jieba.lcut(text, cut_all=False)

"""

# 导包
import jieba    # 中文分词工具


# 1.定义函数，演示 精确模式	将句子最精确地切开	文本分析
def demo01():
    print("jieba分词精确模式")
    # 1.创建输入文本
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2.使用jieba进行分词
    # sentence:输入文本，cut_all:是否使用全模式, 默认False, 也就是精确模式
    words = jieba.cut(sentence=text, cut_all=False)

    # 3.打印分词结果
    print(words)
    # 方法1: list
    # print(list(words))
    # 方法2: for
    # for word in words:
    #     print(word)

    # 4.使用jieba.lcut
    words = jieba.lcut(sentence=text, cut_all=False)
    print(words)
    print("-"*70)
    ...

# 2.定义函数，演示 全模式	扫描所有可成词的词语，速度快，不能消除歧义	关键词提取
def demo02():
    print("jieba分词全模式")
    # 1.创建输入文本
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2.使用jieba进行分词
    # sentence:输入文本，cut_all:是否使用全模式, 默认False, 也就是精确模式
    words = jieba.cut(sentence=text, cut_all=True)

    # 3.打印分词结果
    print(words)
    # 方法1: list
    # print(list(words))
    # 方法2: for
    # for word in words:
    #     print(word)

    # 4.使用jieba.lcut
    words = jieba.lcut(sentence=text, cut_all=True)
    print(words)
    print("-"*70)
    ...
# 3.定义函数，演示 搜索引擎模式	在精确模式基础上对长词再次切分，提高召回率	搜索引擎
"""
解释: 搜索引擎模式, 在精确模式分词的基础上, 对长词进行再次切分, 提高召回率.
例如：
    场景1: 用户录入  "程序员"
        精确模式:       分词结果(程序员)，只能匹配包含完整 "程序员" 的文档.
        搜索引擎模式:    分词结果(程序员，程序，员)，不仅能匹配 "程序员" 的文档, 还能匹配 "程序", "员"的文档. 提高召回率.

    场景2: 实际应用场景(电商搜索),  商品标题为：小米手机保护壳，用户搜索：小米壳
        精确模式：       小米手机保护壳 分词为("小米"，"手机"，"保护壳")，用户搜索切分为(小米，壳),无法匹配
        搜索引擎模式:    小米手机保护壳 分词为("小米手机", "小米"，"手机"，"保护壳"，"保护"，"壳")，用户搜索切分为(小米，壳)，能匹配
"""
def demo03():
    print("jieba分词搜索引擎模式")
    # 1.创建输入文本
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2.使用jieba进行分词
    # sentence:输入文本
    words = jieba.cut_for_search(sentence=text)

    # 3.打印分词结果
    print(words)
    # 方法1: list
    # print(list(words))
    # 方法2: for
    # for word in words:
    #     print(word)

    # 4.使用jieba.lcut_for_search
    words = jieba.lcut_for_search(sentence=text)
    print(words)
    print("-"*70)
    ...

# 4.定义函数，演示 中文繁体分词	支持繁体中文文本分词	中国香港、台湾地区文本
# 5.定义函数，演示 自定义词典	加载用户词典，提升识别准确率；格式：词语 词频 词性（词频、词性可省略）	生僻词/专业术语场景


# 主程序
if __name__ == '__main__':
    demo01()
    demo02()
    demo03()