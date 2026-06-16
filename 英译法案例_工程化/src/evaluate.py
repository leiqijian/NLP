import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from dataset import get_dataloader
from predict import predict_batch
from src.model import MyTransformer
from src.tokenizer.english_tokenizer import EnglishTokenizer
from src.tokenizer.french_tokenizer import FrenchTokenizer


def evaluate(model, test_dataloader,fr_tokenizer):
    model.eval()
    # 存模型生成的翻译（token 序列）
    predictions = []
    # reference_id shape [[*,*,*,*,*],[*,*,*,*],[*,*,*,]]
    # 存测试集的参考答案
    references = []
    # reference shape [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*,]]]
    for inputs, targets in tqdm(test_dataloader, desc="评估"):
        inputs = inputs.to(config.DEVICE)
        # inputs.shape == [batch_size, seq_len]
        targets = targets.tolist()
        # targets:[[sos,*,*,*,*,eos],[sos,*,*,*,eos,pad],[sos,*,*,eos,pad,pad]]

        batch_results = predict_batch(model,inputs,fr_tokenizer)
        # batch_results: [[*,*,*,*,*],[*,*,*,*],[*,*,*,]]

        # 对targets去除特殊符号

        predictions.extend(batch_results) # 把元素一个一个拿出来放入
        references.extend([[target[1:target.index(fr_tokenizer.eos_token_index)]] for target in targets])

    bleu = corpus_bleu(references,predictions)
    return bleu

def run_evaluate():
    #
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    fr_tokenizer = FrenchTokenizer.from_vocab(config.MODEL_DIR / "fr_vocab.txt")

    model = MyTransformer(
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        encoder_num_layers=config.ENCODER_NUM_LAYERS,
        decoder_num_layers=config.DECODER_NUM_LAYERS,
        batch_first=True,
        src_vocab_list=en_tokenizer.vocab_list,
        tgt_vocab_list=fr_tokenizer.vocab_list,
        max_src_len=config.MAX_SRC_LEN,
        max_tgt_len=config.MAX_TGT_LEN,
    )
    model.load_state_dict(torch.load(config.MODEL_DIR / "model.pth", map_location=config.DEVICE))

    # 4.数据集
    test_dataloader = get_dataloader(config.TEST_DATA)

    # 5.评估模型
    bleu = evaluate(model,test_dataloader,fr_tokenizer)
    print("评估结果：")
    print(f"bleu:{bleu}")




if __name__ == '__main__':
    run_evaluate()