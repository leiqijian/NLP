"""
演示
    jieba分词
介绍
    分词：将连续的字序列按照语法规范切分成连续的词序列的过程
    分词之后的最小单元：词/token

常见的分词包：
    jieba: 最流行的python中文分词库，支持精确模式、全模式、搜索引擎模式、中文繁体分词、自定义词典
    IK: 基于JAVA的中文分词工具，ElasticSearch搜索引擎专用
    SnowNLP: 基于概率算法的中文自然语言处理工具，支持情感分析、文本分类
    pyltp: 哈工大推出的python中文分词工具
    THULAC: 清华退出的中文词法分析工具，支持分词和词性标注

jieba分词模式：
    1.多种分词模式
        **精确模式**	将句子最精确地切开	文本分析
        **全模式**	扫描所有可成词的词语，速度快，不能消除歧义	关键词提取
        **搜索引擎模式**	在精确模式基础上对长词再次切分，提高召回率	搜索引擎
    2.**中文繁体分词**	支持繁体中文文本分词	中国香港、台湾地区文本
    3.**自定义词典**	加载用户词典，提升识别准确率；格式：`词语 词频 词性`（词频、词性可省略）	生僻词/专业术语场景
总结：
    上述分词模式，其实就是分词粒度不同

涉及到的API:
    精确模式:       jieba.cut(text, cut_all=False) / jieba.lcut(text, cut_all=False)
    全模式:         jieba.cut(text, cut_all=True) / jieba.lcut(text, cut_all=True)
    搜索引擎模式:   jieba.cut_for_search(text)     / jieba.lcut_for_search(text)
    自定义词典:     jieba.load_userdict('...')

    注: cut() 返回生成器(generator), lcut() 返回列表(list)
需要掌握：
    精确模式，jieba.lcut(text, cut_all=False)

"""
import jieba

# 定义函数，演示 jieba的精确模式分词，适用于 通用文本分析，词性标注，信息提取等
def demo01():
    pass

# 2.定义函数，演示 jieba的全模式分词，适用于 关键词提取
def demo02():
    pass

# 3.定义函数，演示 jieba 搜索引擎模式分词，适用于：搜索引擎，文本匹配
def demo03():
    pass

# 4.定义函数，演示 jieba的中文繁体分词，适用于 对繁体中文进行分词
def demo04():
    pass


if __name__ == '__main__':
    demo01()

