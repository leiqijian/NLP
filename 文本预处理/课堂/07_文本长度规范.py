"""
演示
    文本长度规范，也就是把输入的句子 截断填充到固定长度seq_len
解释：
    一般模型需要输入的张量为相同尺寸，也就是张量的seq_len序列维度的长度要一致
    所以需要把同一个批次的样本句子长度截断填充到相同的长度。

实现方式:
    1.使用python基础实现
    2.使用tensorflow的sequence方法实现，pip install tensorflow
"""
# from tensorflow.keras.preprocessing import sequence

# 定义最大长度
SEQ_LEN = 5

# # 1.使用python基础实现
# def demo01(sentences, seq_len=SEQ_LEN):
#     """
#     输入的一个批次的样本，把它们的句子长度 截断填充到固定长度seq_len
#     :param sentences: 输入的一个批次的样本
#     :param seq_len: 固定长度
#     :return: 截断填充到固定长度的一个批次的样本
#     """
#     # 1.定义一个空列表，保存最终结果
#     results = []
#
#     # 2.遍历批次，对每个样本进行截断填充到固定长度seq_len
#     for sentence in sentences:
#         sentence = (sentence + [0]*seq_len)[:seq_len]
#         results.append(sentence)
#     return results

def demo02(sentences, seq_len=SEQ_LEN):
    results = []

    # 2.遍历批次，对每个样本进行截断填充到固定长度seq_len
    for sentence in sentences:
        # print(sentence)
        # 1.对长句子进行截断
        if len(sentence) >= seq_len:
            sentence = sentence[:seq_len]

        # 2.对短句子进行填充，填充0
        else:
            sentence = sentence + [0]*(seq_len-len(sentence))

        # 3.将处理后的句子添加到results中
        results.append(sentence)
    return results


# 主程序
if __name__ == '__main__':
    sentences = [
        [1,1,2,3,4,5,6],
        [2,1,2,3,4,5,6,8,7,5,9,8,7,5,9,8,7,5,9,8,7,5,9],
        [3,1,2,3,4],
        [4,1,2,3,4,5,6,8,7,5,9,8,7,5,9,8,7,5,9,8,7,5,9,8,7,5,9],
    ]
    result = demo02(sentences)
    print(result)