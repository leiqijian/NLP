"""
演示
    POS词性标注 和 NER命名实体识别
POS(Part-Of-Speech tagging)词性标注:
    标注出一段文本中每个词/token的词性
NER(命名实体识别, Named Entity Recognition):
    就是识别出一段文本中 的 命名实体/专有名词.
    命名实体:
        人名, 地名, 机构名等专有名词
    作用:
        提取关键语义信息。
需要掌握的:
    jieba.posseg.lcut 进行 分词 + POS词性标注

"""

# 导包
import jieba.posseg as pseg

# 1.定义函数，演示POS词性标注
def demo01():
    # 1.创建输入文本
    text = "我爱津安创意园的103教室的可爱的小伙伴们，尤其是帅气的'阳帆'同学"

    # 2.使用pseg.lcut进行 分词 + POS
    res1 = pseg.lcut(text)
    # print(res1)

    # 3.打印POS结果
    for token, tag in res1:
        print(f"{token}: {tag}")
    print("-"*70)
    ...

# 2.定义函数，演示NER命名实体识别
# 思路: 先用pseg进行POS，然后判断哪些token的词性是命名实体（使用词性列表来枚举），最终返回命名实体，注意这是一个简化的NER思路
def demo02(text):
    # 1.使用pseg.lcut进行分词 和 词性标注
    result = pseg.lcut(text)
    # 2.定义命名实体的词性-实体类型的字典
    flag_to_entity = {
        "nr": "PERSON",  # 人名
        "ns": "LOCATION",  # 地名
        "nt": "ORGANIZATION",  # 机构团体名
        "nz": "OTHER",  # 其他专名
        "t": "TIME",  # 时间
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

# 主程序
if __name__ == '__main__':
    demo01()
    text = "何润东，名项羽，字吕布，江湖混名步惊云，此人手无缚鸡之力"
    print(demo02(text))
    ...







