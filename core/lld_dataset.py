import pickle
from os.path import join as pjoin

import numpy  as np
import torch as tc
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .Normalizer import Normalizer


class lld_dataset(Dataset):

    def __init__(self, dataFolder: str, listPath: str, scale=Normalizer()):
        clist = (l.strip().split() for l in open(listPath))
        names, labels = zip(*clist)
        self.samples = [pickle.load(open(pjoin(dataFolder, i), 'rb')) for i in names]
        scale(self.samples)
        self.samples = tuple(map(lambda x: tc.tensor(x, dtype=tc.float32).share_memory_(), self.samples))
        self.labels = tc.tensor(tuple(map(int, labels))).share_memory_()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


def lld_collate_fn(data):
    llds, labels = zip(*data)
    return pad_sequence(llds, True), tc.stack(labels), tc.from_numpy(
        np.fromiter(map(len, llds), np.int16, count=len(llds)))
