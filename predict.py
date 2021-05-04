import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
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
from main import *


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def predict():
    (SRC, TGT), (train_iter, valid_iter) = load_data(
        train_src="dataset/train.bpe.th",
        train_dst="dataset/train.bpe.en",
        valid_src="dataset/valid3k.bpe.th",
        valid_dst="dataset/valid3k.bpe.en",
    )

    pad_idx = TGT.vocab.stoi["<blank>"]

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=2)
    model.load_state_dict(torch.load("weights/model_11.pt"))
    model.eval()

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(
            model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"]
        )
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        break


if __name__ == "__main__":
    predict()
