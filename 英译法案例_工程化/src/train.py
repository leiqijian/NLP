import time

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from src.tokenizer.english_tokenizer import EnglishTokenizer
from src.tokenizer.french_tokenizer import FrenchTokenizer
from src.model import MyTransformer
import config
from torch import nn
from tqdm import tqdm


def train():
    # 加载训练数据
    train_loader = get_dataloader(str=config.TRAIN_DATA)

    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR/'en_vocab.txt')
    fr_tokenizer = FrenchTokenizer.from_vocab(config.MODEL_DIR/'fr_vocab.txt')

    # 创建模型
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
        max_tgt_len=config.MAX_TGT_LEN
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    best_loss = float("inf")
    model = model.to(config.DEVICE)
    for epoch in range(config.EPOCHS):
        model.train()

        loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        print(f"epoch{epoch}")
        print(f"Loss:{loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR/'model.pth')
            print("保存模型")

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    total_loss = 0
    for idx, (source_tensor, target_tensor) in enumerate(tqdm(train_loader)):
        encoder_input = source_tensor.to(config.DEVICE)
        decoder_input = target_tensor[:, :-1].to(config.DEVICE)
        decoder_targets = target_tensor[:, 1:].to(config.DEVICE)

        src_padding_mask = (encoder_input == model.src_pad_idx).to(config.DEVICE)
        tgt_padding_mask = (decoder_input == model.tgt_pad_idx).to(config.DEVICE)
        tgt_causal_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(config.DEVICE)

        decoder_outputs = model(
            src=encoder_input,
            tgt=decoder_input,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_causal_mask
        )

        loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.shape[-1]), decoder_targets.reshape(-1))
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


if __name__ == '__main__':
    train()



