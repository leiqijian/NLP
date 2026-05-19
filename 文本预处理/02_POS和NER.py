"""
演示
    POS词性标注 和 NER命名实体识别
解释：
    POS(词性标注 Part Of Speech Tagging), 标注文本中每个 词/token 的词性.
        词性：以 语法 为主要依据，把词划分为不同的类别，比如：动词v，名称n，形容词a
    NER(命名实体识别, Named Entity Recognition), 就是识别出一段文本中 的 命名实体/专有名词.
        命名实体:
            人名, 地名, 机构名等专有名词
        作用：
            提取关键语义信息。(了解概念即可)
需要掌握的：
    jieba.posseg.lcut进行POS词性标注

"""
# 导包
import jieba.posseg as pseg

# 演示POS
# 1.定义输入文本
text = "我爱深圳创维创新谷5层506教师的可爱的小伙伴,尤其是欢快的‘胡欢’同学"
# 2.使用pseg.lcut进行分词 和 词性标注
result1 = pseg.lcut(text)
# print(result1)
# 3.打印词性标注结果，token-flag
for token, flag in result1:
    print(f"{token}:{flag}")

# 4.进行NER命名实体识别
def extract_entities(text):
    """
    思路：根据jieba.posseg的词性标注结果来提取专有名称，得到命名实体
    注意，这是一个简化的NER思路
    """
    # 1.使用pseg.lcut进行分词 和 词性标注
    result = pseg.lcut(text)
    # 2.定义命名实体的词性-实体类型的字典
    flag_to_entity = {
        "nr": "PERSON", # 人名
        "ns": "LOCATION", # 地名
        "nt": "ORGANIZATION", # 机构团体名
        "nz": "OTHER", # 其他专名
        "t": "TIME",    # 时间
        "m": "NUMBER",  # 数字
    }
    # 3.判断分词结果中有哪些命名实体
    result_entity = []
    for token, flag in result:
        # 1.获取字典中flag对应的实体类型
        entity = flag_to_entity.get(flag, None)
        if entity is not None:
            result_entity.append((token, entity))
    return result_entity

# 进行NER命名实体识别
print(extract_entities(text))