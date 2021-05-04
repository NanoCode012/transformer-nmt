import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import os

from encoder import *
from decoder import *
from nmt import *
from multiheaded_attention import *
from utils import *
from generator import *
from critirion import *
from optimizer import *
from dataset import *

use_cuda = True


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    global use_cuda
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        data.requires_grad = False
        src = data
        tgt = copy.deepcopy(data)

        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()

        yield Batch(src, tgt, 0)


max_src_in_batch, max_tgt_in_batch = None, None


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(
                "Epoch Step: %d Loss: %f Tokens per Sec: %f"
                % (i, loss / batch.ntokens, tokens / elapsed)
            )
            start = time.time()
            tokens = 0
        torch.cuda.empty_cache()
    return total_loss / total_tokens


def main():
    global use_cuda

    (SRC, TGT), (train_iter, valid_iter) = load_data(
        train_src="dataset/train.bpe.th",
        train_dst="dataset/train.bpe.en",
        valid_src="dataset/valid3k.bpe.th",
        valid_dst="dataset/valid3k.bpe.en",
    )

    pad_idx = TGT.vocab.stoi["<blank>"]
    BATCH_SIZE = 60000
    # train_iter = MyIterator(
    #     train,
    #     batch_size=BATCH_SIZE,
    #     device=0,
    #     repeat=False,
    #     sort_key=lambda x: (len(x.src), len(x.trg)),
    #     batch_size_fn=batch_size_fn,
    #     train=True,
    # )
    # valid_iter = MyIterator(
    #     val,
    #     batch_size=BATCH_SIZE,
    #     device=0,
    #     repeat=False,
    #     sort_key=lambda x: (len(x.src), len(x.trg)),
    #     batch_size_fn=batch_size_fn,
    #     train=False,
    # )

    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=0, smoothing=0.0)
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=2)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    model_opt = NoamOpt(
        model.src_embed[0].d_model, factor=1, warmup=400, optimizer=optimizer
    )

    if not os.path.exists("weights"):
        os.mkdir("weights")

    for epoch in range(1000):
        print("Entering epoch : %d" % epoch)
        model.train()
        run_epoch(
            (rebatch(pad_idx, b) for b in train_iter),
            model,
            SimpleLossCompute(model.generator, criterion, model_opt),
        )
        model.eval()
        print(
            run_epoch(
                (rebatch(pad_idx, b) for b in valid_iter),
                model,
                SimpleLossCompute(model.generator, criterion, None),
            )
        )

        torch.save(model.state_dict(), f"weights/model_{epoch}.pt")


if __name__ == "__main__":
    print(torch.__version__)  # 1.7+
    main()
