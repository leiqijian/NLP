import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.tokenizer.english_tokenizer import EnglishTokenizer
from src.tokenizer.french_tokenizer import FrenchTokenizer

# def split_data(data_path=DATA_PATH,train_ratio=0.8,val_ratio=0.1):
#     # 1.读取文件
#     with open(data_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#     # 2.切分src-tgt,构成src-tgt句子对列表
#     pairs = [line.strip().split("\t") for line in lines]
#     pairs = [(clean_text(src), clean_text(tgt)) for src, tgt in pairs]
#     # 3.去重，一定要去重，否则会造成严重的数据泄露
#     new_pairs = []
#     for pair in pairs:
#         if pair not in new_pairs:
#             new_pairs.append(pair)
#     pairs = new_pairs
#     # 4.拆分数据集
#     # 拆分为训练集和测试集
#     train_pairs, valid_test_pairs = train_test_split(pairs, train_size=train_ratio, random_state=2)
#     # 拆分测试集为验证集和测试集
#     valid_pairs, test_pairs = train_test_split(valid_test_pairs, train_size=val_ratio/(1-train_ratio), random_state=2)
#     # 5.返回结果
#     return train_pairs, valid_pairs, test_pairs

def data_preprocessing():
    # todo: 1.加载数据
    print("开始处理数据")
    df = pd.read_csv(config.RAW_DATA_PATH, sep='\t',
                header=None, usecols=[0, 1],names=['en', 'fr'],encoding= 'utf-8').dropna()
    # print(df.head())

    # 去重：en和fr都相同的行只保留第一条
    before = len(df)
    df = df.drop_duplicates(subset=['en', 'fr']).reset_index(drop=True)
    print(f"去重: {before} -> {len(df)} (移除 {before - len(df)} 条)")

    # todo：2.划分数据集
    # 拆分为训练集和测试集 (8:2)
    train_data, valid_test_data = train_test_split(df, test_size=0.2, random_state=config.RANDOM_SEED)
    # 拆分测试集为验证集和测试集
    valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, random_state=config.RANDOM_SEED)

    # print(train_data['en'].tolist())

    # todo: 3.构建词表
    # 构建词表索引
    EnglishTokenizer.build_vocab(train_data['en'].tolist(), config.MODEL_DIR/'en_vocab.txt')
    FrenchTokenizer.build_vocab(train_data['fr'].tolist(), config.MODEL_DIR/'fr_vocab.txt')

    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR/'en_vocab.txt')
    fr_tokenizer = FrenchTokenizer.from_vocab(config.MODEL_DIR/'fr_vocab.txt')

    # 把原句子映射成索引集
    train_data['fr'] = train_data['fr'].apply(lambda x : fr_tokenizer.encode(x,add_sos_eos=True)) # encoder input
    train_data['en'] = train_data['en'].apply(lambda x : en_tokenizer.encode(x,add_sos_eos=False)) # decoder input

    valid_data['fr'] = valid_data['fr'].apply(lambda x : fr_tokenizer.encode(x,add_sos_eos=True))
    valid_data['en'] = valid_data['en'].apply(lambda x : en_tokenizer.encode(x,add_sos_eos=False))

    test_data['fr'] = test_data['fr'].apply(lambda x : fr_tokenizer.encode(x,add_sos_eos=True))
    test_data['en'] = test_data['en'].apply(lambda x : en_tokenizer.encode(x,add_sos_eos=False))

    train_data.to_json(config.PROCESSED_DATA_DIR/'train.jsonl', orient='records', lines=True)
    valid_data.to_json(config.PROCESSED_DATA_DIR/'valid.jsonl', orient='records', lines=True)
    test_data.to_json(config.PROCESSED_DATA_DIR/'test.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    data_preprocessing()