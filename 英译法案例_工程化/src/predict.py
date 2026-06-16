import torch
import config
import jieba

from src.model import MyTransformer
from src.tokenizer.english_tokenizer import EnglishTokenizer
from src.tokenizer.french_tokenizer import FrenchTokenizer

@torch.no_grad()
def predict_batch(model,inputs,fr_tokenizer):
    model.eval()
    model = model.to(config.DEVICE)

    # 编码
    src_pad_mask = (inputs == model.src_pad_idx).to(config.DEVICE)  # True=PAD位置，需要被忽略
    memory = model.encoder(inputs, src_pad_mask)  # encoder 内部已做 .to，无需重复

    # 解码
    batch_size = inputs.shape[0]
    decoder_input = torch.full(size=[batch_size, 1], fill_value=fr_tokenizer.sos_token_index,
                               dtype=torch.long, device=config.DEVICE)

    # 存储结果
    generated = []
    # 标记是否完成
    is_finished = torch.full(size=(batch_size,), fill_value=False, device=config.DEVICE)

    # 自回归生成
    for i in range(config.MAX_SRC_LEN):
        # 每步更新 tgt_pad_mask（基于当前 decoder_input）
        tgt_pad_mask = (decoder_input == model.tgt_pad_idx).to(config.DEVICE)
        # 因果掩码
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(config.DEVICE)
        decoder_output = model.decoder(
            encoder_output=memory,
            tgt=decoder_input,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask)
        # decoder_output shape [batch_size,tgt_seq_len,en_vocab_size]
        # 保存预测结果
        next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)  # 取最后的一个时间步的最大概率
        # next_token_indexes = next_token_indexes.unsqueeze(1) # 扩展维度
        # next_token_indexes.shape [batch_size,1]

        # 更新隐藏状态
        generated.append(next_token_indexes)
        # 更新输入 拼接
        decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)
        # 判断是否应该结束
        is_finished |= (next_token_indexes.squeeze(1) == fr_tokenizer.eos_token_index)
        if is_finished.all():
            break
    # 处理预测结果
    # generated shape [tensor([batch_size,1])]
    generated_tensor = torch.cat(generated, dim=-1)
    # generated_tensor.shape [batch_size,seq_len]
    generated_list = generated_tensor.tolist()
    # 去掉eos之后的token_id
    for index, sentence in enumerate(generated_list):
        if fr_tokenizer.eos_token_index in sentence:
            eos_pos = sentence.index(fr_tokenizer.eos_token_index)
            generated_list[index] = sentence[:eos_pos]
    return generated_list

def predict(text,model,fr_tokenizer,en_tokenizer):
    # 1.处理输入获取句子索引
    indexes = en_tokenizer.encode(text)
    
    input_tensor = torch.tensor([indexes],dtype=torch.long) # 输入的维度是 [batch_size, seq_len]
    input_tensor = input_tensor.to(config.DEVICE)
    # 2.预测
    batch_result = predict_batch(model,input_tensor,fr_tokenizer)

    return fr_tokenizer.decode(batch_result[0])

def run_predict():
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


    print("欢迎使用中英翻译系统(输入q或者quit退出)")
    while True:
        user_input = input("英文:")
        if user_input == "q" or user_input == "quit":
            print("欢迎下次再来")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue

        result  = predict(user_input,model,fr_tokenizer,en_tokenizer)
        print("翻译结果：",result)

if __name__ == '__main__':
    run_predict()
