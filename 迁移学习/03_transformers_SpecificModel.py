"""
演示
    huggingface 的 transformers库 如何使用BertModel具体模型方法来实现常见的NLP任务，
    包括 文本分类，特征提取，完形填空，阅读理解(问答任务)，文本摘要，NER命名实体识别

需要准备的工作：
    1.安装transformers datasets, pip install transformers datasets
    2.复制预训练的模型文件到自己电脑
    3.复制数据文件

需要掌握的:
    具体模型:
        # 1.加载模型和分词器tokenizer
        my_model = BertModel.from_pretrained(
            r"model/bert-base-chinese",
            local_files_only=True,
        )
        my_tokenizer = BertTokenizer.from_pretrained(
            r"model/bert-base-chinese",
            local_files_only=True,
        )
        # 2.tokenizer编码，获取输入文本的索引序列，文本 -> 分词 -> token序列 -> 索引序列
        inputs = tokenizer(
            **kwargs,   # 其他参数，比如text
            return_tensors="pt",   # 返回pytorch张量
            padding=True,   # 是否填充，填充到当前批次的最大长度
            truncation=True,   # 是否截断，截断到最大长度
            max_length=256, # 最大长度
        )
        # inputs通常是一个字典，通常键为：input_ids, attention_mask, token_type_ids
        # 3.输入到模型
        outputs = model(**inputs)

"""

# 导包
import torch
# 导入transformers库中的具体模型和分词器相关类
from transformers import (
    BertConfig,  # 具体配置
    BertTokenizer,  # 具体分词器
    BertModel,  # 具体模型
    BertForSequenceClassification,  # 具体模型，序列分类
    BertForMaskedLM,  # 具体模型，掩码语言模型，用于 完形填空
    BertForQuestionAnswering,  # 具体模型，问答/阅读理解
    # BertForSeq2SeqLM,  # 具体模型，序列到序列语言模型，用于 文本摘要/文本生成
    BertForTokenClassification  # 具体模型，命名实体识别NER/词性标注POS
)

from transformers import BartForConditionalGeneration, BartTokenizer    # 序列到序列语言模型，用于 文本摘要/文本生成

# 1.定义通用函数，实现transformers的BertModel方法 的推理过程，模拟上线效果
@torch.no_grad() # 关闭梯度计算
def demo_Bertmodel(model, tokenizer, **kwargs):
    """
    这是一个通用的推理函数，支持各种 Bertmodel, 具体处理 输入文本/数据，返回结果
    :param model: 模型路径 or 模型名称(重新下载)
    :param tokenizer: 分词器对象，内部自带了词表
    :param kwargs: 其他参数，比如text
    :return: 模型输出，一般是字典形式，包括logits
    """
    # 1.tokenizer编码，获取输入文本的索引序列(token id)
    # 文本 -> 分词 -> token序列 -> 索引序列
    inputs = tokenizer(
        **kwargs,   # 其他参数，比如text
        return_tensors="pt",   # 输入张量类型，pt-pytorch张量, tf-tensorflow张量
        padding=True, # 是否填充，填充到当前批次的最大长度
        truncation=True, # 是否截断，截断到最大长度max_length
        max_length=256, # 最大长度
    )

    # inputs通常是一个字典，通常键为：input_ids(token id), attention_mask(注意力填充掩码), token_type_ids(句子id)
    # 值的形状都是BS

    # 2.输入到模型，进行推理
    outputs = model(**inputs)
    return outputs, inputs

# 主程序
if __name__ == '__main__':
    # todo 1.文本分类/情感分析，chinese_sentiment
    # 1.加载模型和分词器tokenizer
    my_model = BertForSequenceClassification.from_pretrained(
        r"./model/chinese_sentiment",
        local_files_only=True, # 模型是否只从本地加载,不要下载模型
    )
    my_tokenizer = BertTokenizer.from_pretrained(
        r"./model/chinese_sentiment",
        local_files_only=True,
    )
    # print(f"my_model: {my_model}")
    # print(f"my_tokenizer: {my_tokenizer}")

    # 2.定义输入数据text
    text = [
        "我非常喜欢这个电影",
        "我非常不喜欢这个电影",
        "今天一航同学穿了黄色衣服，惨不忍睹，下次别穿了"
    ]

    # 3.模型推理
    outputs, inputs = demo_Bertmodel(my_model, my_tokenizer, text=text)
    # print(f"outputs.logits: {outputs.logits},shape: {outputs.logits.shape}") # (B, num_classes)
    # print(f"inputs: {inputs}, input_ids.shape: {inputs['input_ids'].shape}")

    # 4.获取预测结果
    preds = torch.argmax(outputs.logits, dim=-1) # (B,)
    print(f"preds: {preds}")
    print(f"最终预测结果: star {(preds+1).tolist()}")
    print("="*70)

    # # todo 2.特征提取，bert-base-chinese
    # # 1.加载模型和分词器tokenizer
    # my_model = BertModel.from_pretrained(
    #     r"./model/bert-base-chinese",
    #     local_files_only=True,  # 模型是否只从本地加载,不要下载模型
    # )
    # my_tokenizer = BertTokenizer.from_pretrained(
    #     r"./model/bert-base-chinese",
    #     local_files_only=True,
    # )
    #
    # # 2.定义输入数据text
    # text = [
    #     "我非常喜欢这个电影",
    #     "我非常不喜欢这个电影",
    #     "今天一航同学穿了黄色衣服，惨不忍睹，下次别穿了"
    # ]
    #
    # # 3.模型推理
    # outputs, inputs = demo_Bertmodel(my_model, my_tokenizer, text=text)
    # print(f"outputs.last_hidden_state: {outputs.last_hidden_state}, shape: {outputs.last_hidden_state.shape}") # BSD
    # print(f"outputs.pooler_output: {outputs.pooler_output}, shape: {outputs.pooler_output.shape}") # BD
    # # outputs.last_hidden_state: BSD, 所有时间步的隐藏状态，代表每个token的语义表示，用于token级的任务，比如 NER/POS
    # # outputs.pooler_output: BD, 池化输出，代表整个句子的语义表示，用于句子级的任务，比如 文本分类
    # print("=" * 70)

    # # todo 3.完形填空任务，model/chinese-bert-wwm
    # # 1.加载模型和分词器tokenizer
    # my_model = BertForMaskedLM.from_pretrained(
    #     r"model/chinese-bert-wwm",
    #     local_files_only=True,
    # )
    # my_tokenizer = BertTokenizer.from_pretrained(
    #     r"model/chinese-bert-wwm",
    #     local_files_only=True,
    # )
    #
    # # 2.定义输入数据text
    # text = [
    #     "床前明月[MASK]，疑是地上霜",
    #     "停车做爱[MASK]林晚",
    # ]
    # # 3.模型推理，得到output,里面有logits
    # outputs, inputs = demo_Bertmodel(my_model, my_tokenizer, text=text)
    # # outputs.logits: (B,S,V), logits表示每个token的预测分数
    # print(outputs.logits.shape)
    # preds = torch.argmax(outputs.logits, dim=-1)  # (B,S)
    # print(preds)
    # print(preds[[0, 1], [5, 5]])
    # print(my_tokenizer.convert_ids_to_tokens(preds.reshape(-1)))
    # print("-" * 40)

    # # todo 4.文本摘要，model/distilbart-cnn-12-6
    # # 1.加载模型和分词器tokenizer
    # my_model = BartForConditionalGeneration.from_pretrained(
    #     r"model/distilbart-cnn-12-6",
    #     local_files_only=True,
    # )
    # my_tokenizer = BartTokenizer.from_pretrained(
    #     r"model/distilbart-cnn-12-6",
    #     local_files_only=True,
    # )
    # # print(my_model)
    # # print(my_tokenizer)
    # # 2.定义输入数据text
    # text = "BERT is a transformers model pretrained on a large corpus of English data " \
    #        "in a self-supervised fashion. This means it was pretrained on the raw texts " \
    #        "only, with no humans labelling them in any way (which is why it can use lots " \
    #        "of publicly available data) with an Bertmatic process to generate inputs and " \
    #        "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
    #        "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
    #        "words in the input then run the entire masked sentence through the model and has " \
    #        "to predict the masked words. This is different from traditional recurrent neural " \
    #        "networks (RNNs) that usually see the words one after the other, or from Bertregressive " \
    #        "models like GPT which internally mask the future tokens. It allows the model to learn " \
    #        "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
    #        " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
    #        "sentences that were next to each other in the original text, sometimes not. The model then " \
    #        "has to predict if the two sentences were following each other or not."
    # # 3.模型推理，得到output,里面有logits
    # outputs, inputs = demo_Bertmodel(my_model, my_tokenizer, text=text)
    # # outputs.logits: (B,S,V), logits表示每个token的预测分数
    # print(outputs.logits.shape)  # (1, 16, 50264)
    # # 4.获取预测结果
    # preds = torch.argmax(outputs.logits, dim=-1)  # (B,S)
    # # print(preds)
    # print(my_tokenizer.convert_ids_to_tokens(preds.reshape(-1)))
    # print(my_tokenizer.decode(preds.reshape(-1)))
    # print("-" * 40)

    # # todo 5.NER命名实体识别, token级别任务，model/roberta-base-finetuned-cluener2020-chinese
    # # 1.加载模型和分词器tokenizer
    # my_model = BertForTokenClassification.from_pretrained(
    #     r"model/roberta-base-finetuned-cluener2020-chinese",
    #     local_files_only=True,
    # )
    # my_tokenizer = BertTokenizer.from_pretrained(
    #     r"model/roberta-base-finetuned-cluener2020-chinese",
    #     local_files_only=True,
    # )
    # my_config = BertConfig.from_pretrained(
    #     r"model/roberta-base-finetuned-cluener2020-chinese",
    #     local_files_only=True,
    # )
    #
    # # 2.定义输入数据text
    # text = "我叫阳帆，在中山大学读博士，毕业以后留在了食堂当保安，现在有个对象，是新东方的厨师"
    # # 3.模型推理，得到output,里面有logits
    # outputs, inputs = demo_Bertmodel(my_model, my_tokenizer, text=text)
    # # outputs.logits: (B,S,num_classes), logits表示每个token的预测分数
    # print(outputs.logits.shape)  # (1, 44, 32)
    # # 4.获取预测结果
    # preds = torch.argmax(outputs.logits, dim=-1)[0]
    # print(preds)
    # results = []
    # for value in preds:
    #     results.append(my_config.id2label[value.item()])
    # print(results)

    # todo 6.阅读理解任务，问答式任务，model/chinese_pretrain_mrc_roberta_wwm_ext_large
    # 1.加载模型和分词器tokenizer
    my_model = BertForQuestionAnswering.from_pretrained(
        r"model/chinese_pretrain_mrc_roberta_wwm_ext_large",
        local_files_only=True,
    )
    my_tokenizer = BertTokenizer.from_pretrained(
        r"model/chinese_pretrain_mrc_roberta_wwm_ext_large",
        local_files_only=True,
    )
    # 2.定义输入数据
    context = "《三体2: 黑暗森林》是刘慈欣的科幻小说，是三体系列的第2部。主人公有罗辑、三体人"
    questions = [
        "三体2的作者是谁？",
        "三体2的主人公有哪些？",
    ]

    # 3.模型推理，得到output,inputs
    output, inputs = demo_Bertmodel(
        my_model,
        my_tokenizer,
        text=questions,  # 问题列表
        text_pair=[context] * len(questions)  # 参考文献列表
    )
    print(output.start_logits.shape, output.end_logits.shape)
    # 4.获取预测结果
    # 获取答案的开始索引
    # 获取答案的结束索引
    start_idx = torch.argmax(output.start_logits, dim=-1)
    end_idx = torch.argmax(output.end_logits, dim=-1)
    print(start_idx, end_idx)
    for i, q in enumerate(questions):
        answer = my_tokenizer.decode(inputs.input_ids[i][start_idx[i]:end_idx[i] + 1])
        print(f"问题：{q}")
        print(f"答案：{answer}")
    ...