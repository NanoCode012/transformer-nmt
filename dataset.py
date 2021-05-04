# For data loading.
from torchtext import data, datasets
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
    max_len=1000,
    batch_size=14000,
    device=None,
    train=True,
):
    import pandas as pd
    import os

    raw_data = {"src": [line for line in SRC_DATA], "trg": [line for line in TGT_DATA]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df["src"].str.count(" ") < max_len) & (df["trg"].str.count(" ") < max_len)
    df = df.loc[mask]

    df.to_csv("temp.csv", index=False)

    data_fields = [("src", SRC), ("trg", TGT)]
    dataset = data.TabularDataset("./temp.csv", format="csv", fields=data_fields)

    dataloader = MyIterator(
        dataset,
        batch_size=batch_size,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=train,
        shuffle=True,
    )

    os.remove("temp.csv")

    return dataloader, (dataset if train else None)


def load_data(train_src, train_dst, valid_src, valid_dst):

    from main import batch_size_fn

    SRC, TGT = _create_fields()

    TRAIN_SRC_DATA, TRAIN_TGT_DATA = _read_data(train_src, train_dst)
    train, _data = _create_dataset(
        SRC,
        TGT,
        TRAIN_SRC_DATA,
        TRAIN_TGT_DATA,
        batch_size_fn=batch_size_fn,
        device="cuda:0",
        train=True,
    )

    VALID_SRC_DATA, VALID_TGT_DATA = _read_data(valid_src, valid_dst)
    valid, _ = _create_dataset(
        SRC,
        TGT,
        VALID_SRC_DATA,
        VALID_TGT_DATA,
        batch_size_fn=batch_size_fn,
        device="cuda:0",
        train=False,
    )

    SRC.build_vocab(_data)
    TGT.build_vocab(_data)

    return ((SRC, TGT), (train, valid))


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
