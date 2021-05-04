# For data loading.
from torchtext import data, datasets
from torchtext.legacy.data import Dataset
from main import use_cuda
from utils import subsequent_mask


def _read_data(src, tgt):
    src_data = open(src).read().strip().split("\n")
    tgt_data = open(tgt).read().strip().split("\n")

    return (src_data, tgt_data)


def _create_fields():
    import spacy

    if use_cuda:
        spacy.prefer_gpu()

    spacy_th = spacy.blank("th")
    spacy_en = spacy.load("en_core_web_trf")

    def tokenize_th(text):
        return [tok.text for tok in spacy_th.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = "<s>"
    EOS_WORD = "</s>"
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_th, pad_token=BLANK_WORD)
    TGT = data.Field(
        tokenize=tokenize_en,
        init_token=BOS_WORD,
        eos_token=EOS_WORD,
        pad_token=BLANK_WORD,
    )

    return (SRC, TGT)


def _create_dataset(
    SRC,
    TGT,
    SRC_DATA,
    TGT_DATA,
    batch_size_fn,
    max_len=50,
    batch_size=2000,
    device=None,
):
    import pandas as pd
    import os

    raw_data = {"src": [line for line in SRC_DATA], "tgt": [line for line in TGT_DATA]}
    df = pd.DataFrame(raw_data, columns=["src", "tgt"])

    mask = (df["src"].str.count(" ") < max_len) & (df["tgt"].str.count(" ") < max_len)
    df = df.loc[mask]

    df.to_csv("temp.csv", index=False)

    data_fields = [("src", SRC), ("tgt", TGT)]
    train = data.TabularDataset("./temp.csv", format="csv", fields=data_fields)

    train_iter = MyIterator(
        train,
        batch_size=batch_size,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=True,
        shuffle=True,
    )

    os.remove("temp.csv")

    SRC.build_vocab(train)
    TGT.build_vocab(train)

    return train_iter


def load_data():

    SRC_DATA, TGT_DATA = _read_data("dataset/train.bpe.th", "train.bpe.en")
    SRC, TGT = _create_fields()

    from main import batch_size_fn

    train = _create_dataset(
        SRC, TGT, SRC_DATA, TGT_DATA, batch_size_fn=batch_size_fn, device="0"
    )

    return ((SRC, TGT), (train))


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)

            if use_cuda:
                self.trg_mask = self.trg_mask.cuda()

            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:

            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,
                        self.batch_size_fn,
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
