import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

class TextDataset(Dataset):
    def __init__(self, path, config):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = ByteLevelBPETokenizer("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
        self.tokens = tokenizer.encode(text).ids
        self.seq_len = config.seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_len])
        return x
