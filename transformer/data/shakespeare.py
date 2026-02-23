import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformer.model.transformer_model import Transformer
from transformer.generate import generate
import re

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_words(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)


def build_vocab(text: str, extra_tokens=("<PAD>", "<BOS>", "<EOS>", "<UNK>")):
    words = sorted(list(set(tokenize_words(text))))
    itos = list(extra_tokens) + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict):
    unk_id = stoi["<UNK>"]
    tokens = tokenize_words(text)
    return torch.tensor([stoi.get(tok, unk_id) for tok in tokens], dtype=torch.long)


def decode(ids, itos: list):
    tokens = [itos[i] for i in ids]
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    return text


class ShakespearePrefix2CompletionDataset(Dataset):
    """
    block: 连续 block_size 个 token
    src: block 的前 prefix_len 个 token
    tgt: [BOS] + block + [EOS]
    """
    def __init__(self, text_ids: torch.Tensor, block_size: int, prefix_len: int, bos_id: int, eos_id: int):
        assert text_ids.dim() == 1
        assert 1 <= prefix_len < block_size

        self.text_ids = text_ids
        self.block_size = block_size
        self.prefix_len = prefix_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.n = (len(text_ids) - block_size)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        block = self.text_ids[idx : idx + self.block_size].clone()  # (block_size,)

        src = block[: self.prefix_len]  # (prefix_len,)

        tgt = torch.cat(
            [
                torch.tensor([self.bos_id], dtype=torch.long),
                block,
                torch.tensor([self.eos_id], dtype=torch.long),
            ],
            dim=0,
        )  # (block_size+2,)

        return {"src": src, "tgt": tgt}

def make_collate_fn(pad_id: int):
    def collate(batch):
        B = len(batch)
        src_lens = [x["src"].numel() for x in batch]
        tgt_lens = [x["tgt"].numel() for x in batch]
        S = max(src_lens)
        T = max(tgt_lens)

        src_ids = torch.full((B, S), pad_id, dtype=torch.long)
        tgt_ids = torch.full((B, T), pad_id, dtype=torch.long)

        for i, x in enumerate(batch):
            s = x["src"]
            t = x["tgt"]
            src_ids[i, : s.numel()] = s
            tgt_ids[i, : t.numel()] = t

        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

    return collate



def prepare_vocab(text_path):
    text = load_text(text_path)
    stoi, itos = build_vocab(text)
    vocab_size = len(itos)
    pad_id = stoi["<PAD>"]
    bos_id = stoi["<BOS>"]
    eos_id = stoi["<EOS>"]
    text_ids = encode(text,stoi)
    return stoi,itos,vocab_size,pad_id,bos_id,eos_id,text_ids

def make_shake_dataloader(data_cfg):
    text_ids = data_cfg["text_ids"]
    block_size = data_cfg["block_size"]
    prefix_len = data_cfg["prefix_len"]
    bos_id = data_cfg["bos_id"]
    eos_id = data_cfg["eos_id"]
    pad_id = data_cfg["pad_id"]
    batch_size = data_cfg["batch_size"]
    val_ratio = data_cfg.get("val_ratio", 0.1)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)
    ds = ShakespearePrefix2CompletionDataset(
        text_ids,
        block_size=block_size,
        prefix_len=prefix_len,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    n = len(ds)
    n_val = int(n*val_ratio)
    n_train = n- n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
                      collate_fn=make_collate_fn(pad_id),drop_last=True,)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,
                      collate_fn=make_collate_fn(pad_id),drop_last=False,)
    return  train_loader,val_loader