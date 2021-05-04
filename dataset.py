# For data loading.
from torchtext import data, datasets
from main import use_cuda
from utils import subsequent_mask


def load_data():
    import spacy

    spacy_th = spacy.load("th")
    spacy_en = spacy.load("en")

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

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=(".th", ".en"),
        fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)["src"]) <= MAX_LEN
        and len(vars(x)["trg"]) <= MAX_LEN,
    )
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return ((SRC, TGT), (train, val, test))


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
