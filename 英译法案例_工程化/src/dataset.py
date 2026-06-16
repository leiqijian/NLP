import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config

class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path,orient="records",lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    # 把第 index 条 JSON 记录中的英法文数据，转成 PyTorch 张量
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]["en"], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]["fr"], dtype=torch.long)
        return input_tensor, target_tensor

# todo : zip方法
def collate_fn(batch):
    """
    将不等长的序列 padding 到 batch 内最长长度，如果编辑为等长，构建dataloader的时候会报错
    """
    inputs, targets = zip(*batch)
    # padding_value=0 对应 <pad> 的索引
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


def get_dataloader(str):
    path = config.PROCESSED_DATA_DIR / f"{str}.jsonl"
    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    train_loader = get_dataloader(str=config.TRAIN_DATA)
    valid_loader = get_dataloader(str=config.VALID_DATA)
    test_loader = get_dataloader(str=config.TEST_DATA)
    print("训练集大小",len(train_loader))
    print("验证集大小",len(valid_loader))
    print("测试集大小",len(test_loader))


    for  inputs,targets in train_loader:
        print("输入",inputs.shape) # (batch_size,seq_len)
        print("目标",targets.shape) # (batch_size,)
        break
