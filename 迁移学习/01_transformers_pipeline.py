"""
演示
    huggingface 的 transformers库 如何使用pipeline来实现常见的NLP任务，
    包括 文本分类，特征提取，完形填空，阅读理解(问答任务)，文本摘要，NER命名实体识别

需要准备的工作：
    1.安装transformers datasets, pip install transformers datasets
    2.复制预训练的模型文件到自己电脑
    3.复制数据文件

注意:
    transformers 5.x 不支持pipeline的task=summarization, 无法完成pipeline的生成摘要任务

需要掌握的:
    # 1.实例化pipeline对象
    my_pipeline = pipeline(
        task=task,  # 任务类型
        model=model,    # 模型路径 or 模型名称(重新下载)
        framework="pt", # 模型框架, pt-pytorch, tf-tensorflow
    )
    # 2.送入文本text进行处理
    result = my_pipeline(text, **kwargs)

"""
# 导包
import torch
from transformers import pipeline   # pipeline方法，提供各种预训练模型的pipeline使用方式

# 1.定义函数，实现transformers的pipeline方法
def demo_pipeline(task, model, text=None, **kwargs):
    """
    这个函数实现transformers的 pipeline 方法，来完成各种NLP任务：
    文本分类，特征提取，完形填空，阅读理解(问答任务)，文本摘要，NER命名实体识别
    :param task: 任务类型，比如 sentiment-analysis, feature-extraction, fill-mask, question-answering, summarization
    :param model: 模型路径 or 模型名称(重新下载)
    :param text: 输入文本 或 数据
    :param kwargs: 其他参数
    :return: 模型输出结果
    """
    # 1.实例化pipeline对象
    my_pipeline = pipeline(
        task=task,  # 模型任务类型
        model=model,    # 模型路径 or 模型名称(重新下载)
        # framework="pt", # 模型框架, pt-pytorch, tf-tensorflow
    )

    # 2.传入text到pipeline进行处理
    if text:
        result = my_pipeline(text, **kwargs)
    else:
        result = my_pipeline(**kwargs)

    # 3.返回结果
    print(result)
    print("="*70)
    return result

# 测试
if __name__ == '__main__':
    # 1.情感分类/分析任务，判断文本是好评/差评，model/chinese_sentiment
    # 下载方法：在命令行执行以下命令将模型下载到指定目录
    # git clone https://huggingface.co/techthiyanes/chinese_sentiment D:\NLP-深圳黑马\AI大模型开发就业4期\05-课堂代码\day10_12\model
    result = demo_pipeline(
        task="sentiment-analysis",  # 任务类型
        model=r"model/chinese_sentiment",  # 模型路径 or 模型名称(重新下载)
        text="拔飞唱的真好，下次继续唱",
    )

    # 2.特征提取任务，获取文本特征向量，model/bert-base-chinese
    # 下载方法：git clone https://huggingface.co/google-bert/bert-base-chinese
    result = demo_pipeline(
        task="feature-extraction",  # 任务类型
        model=r"model/bert-base-chinese",  # 模型路径 or 模型名称(重新下载)
        text="是我对你承诺了太多"
    )
    print(torch.tensor(result).shape)
    # BERT模型开头添加CLS,结尾添加SEP

    # 3.完形填空任务，预测[MASK]位置的词，model/chinese-bert-wwm
    # 下载方法：git clone https://huggingface.co/hfl/chinese-bert-wwm
    result = demo_pipeline(
        task="fill-mask",  # 任务类型
        model=r"model/chinese-bert-wwm",  # 模型路径 or 模型名称(重新下载)
        text="停车[MASK]爱枫林晚，霜叶红于二月花"
    )

    # # 4.阅读理解任务，问答任务，model/chinese_pretrain_mrc_roberta_wwm_ext_large
    # # 下载方法：git clone https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large
    # result = demo_pipeline(
    #     task="question-answering",  # 任务类型，问-答任务
    #     model=r"model/chinese_pretrain_mrc_roberta_wwm_ext_large",  # 模型路径 or 模型名称(重新下载)
    #     # text={
    #     #     "context":[
    #     #         "孙悟空在蟠桃园吃桃子",
    #     #         "孙悟空在蟠桃园吃桃子,看到七仙女端着盘子过来",
    #     #         "孙悟空在蟠桃园吃桃子,看到七仙女端着盘子过来,然后把七仙女定住,之后拿走了七仙女的桃子",
    #     #     ],
    #     #     "question":[
    #     #         "孙悟空在哪里吃桃子",
    #     #         "孙悟空看到七仙女端着什么",
    #     #         "孙悟空拿走了七仙女的什么",
    #     #     ]
    #     # }
    #     context=[
    #         "孙悟空在蟠桃园吃桃子",
    #         "孙悟空在蟠桃园吃桃子，看到七仙女端着盘子过来",
    #         "孙悟空在蟠桃园吃桃子，看到七仙女端着盘子过来，然后把七仙女定住，之后拿走了七仙女的桃子",
    #     ],
    #     question=[
    #         "孙悟空在哪里吃桃子",
    #         "孙悟空看到七仙女端着什么",
    #         "孙悟空拿走了七仙女的什么",
    #     ]
    # )
    # print("-"*40)
    # # 5.生成摘要任务, model/distilbart-cnn-12-6
    # # 下载模型 git clone https://huggingface.co/sshleifer/distilbart-cnn-12-6
    # result = demo_pipeline(
    #     task="summarization",  # 生成摘要任务
    #     model=r"model/distilbart-cnn-12-6",  # 模型路径 or 模型名称(重新下载)
    #     text=(
    #         "BERT is a transformers model pretrained on a large corpus of English data "
    #         "in a self-supervised fashion. This means it was pretrained on the raw texts "
    #         "only, with no humans labelling them in any way (which is why it can use lots "
    #         "of publicly available data) with an automatic process to generate inputs and "
    #         "labels from those texts."
    #     )
    # )
    # print("-"*40)
    # # 6.NER任务，命名实体识别,model/roberta-base-finetuned-cluener2020-chinese
    # # 下载方法：git clone https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese
    # result = demo_pipeline(
    #     task="ner",  # 命名实体识别任务
    #     model=r"model/roberta-base-finetuned-cluener2020-chinese",  # 模型路径 or 模型名称(重新下载)
    #     text="我叫胡欢，在清华大学读博士，毕业以后留在了人工智能学院当教授，现在有个对象，是北大的"
    # )
    # print("-"*40)
    ...