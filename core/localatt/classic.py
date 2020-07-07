import math

import torch as tc
import torch.jit as jit
import torch.nn as nn
from torch.nn.utils.rnn import *


class Attn(jit.ScriptModule):
    def __init__(self, ncell: int, window_len: int):
        super().__init__()
        self.do = nn.Dropout(0.5)
        self.u = nn.Conv2d(1, 1, (window_len, ncell * 2), padding=(window_len // 2, 0), bias=True)

    @jit.script_method
    def forward(self, inp, lens):
        # type: (Tensor,Tensor) -> Tensor
        count = len(lens)
        out = tc.empty(count, inp.shape[2], device=inp.device)
        for i in range(count):
            c_len = lens[i]
            t = inp[i, 0:c_len]
            alpha = self.u(self.do(t).view(1, 1, c_len, -1)).view(c_len, 1).softmax(0)
            out[i] = (t * alpha).sum(0)
        return out


def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, 0.01, mode='fan_in')


class localatt(nn.Module):
    def __init__(self, featdim: int, nhid: int, ncell: int, nout: int):
        super(localatt, self).__init__()

        self.extract = nn.Sequential(
            nn.Linear(featdim, nhid),
            nn.LayerNorm(nhid),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(0.3),

            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.extract.apply(init_linear)
        self.rnn1 = tc.nn.LSTM(nhid, ncell, 2, batch_first=True, bias=True, bidirectional=True, dropout=0.5)
        for name, param in self.rnn1.named_parameters():
            if 'bias' in name:
                tc.nn.init.uniform_(param, 0.5 - math.sqrt(1 / ncell), 0.5 + math.sqrt(1 / ncell))
        self.attn = Attn(ncell, 5)

        self.combine = nn.Linear(ncell * 2, nout)

    def forward(self, ilp: [tc.Tensor, ]):
        lens = tc.tensor(tuple(map(len, ilp)), dtype=tc.int16)
        for i in range(len(ilp)):
            ilp[i] = self.extract(ilp[i])
        output = self.rnn1(pad_sequence(ilp,True))[0]
        del ilp
        print(output.shape)
        output: tc.Tensor = self.attn(output,lens)
        return self.combine(output)


if __name__ == '__main__':
    a = localatt(10, 10, 10, 10)
