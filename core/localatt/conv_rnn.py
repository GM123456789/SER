import math

import torch as tc
import torch.nn as nn
from torch import jit
from torch.nn.utils.rnn import *

from .StochasticPooling import RandomPool2d


def init_conv(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weights, a=0.01)


class CNN(jit.ScriptModule):
    __constants__ = ['tot', 'indim']

    def __init__(self, featdim):
        super().__init__()
        self.indim, self.tot = featdim, 14 * 48
        lrelu, selu = nn.LeakyReLU(inplace=True), nn.SELU(True)
        do2, pool =  nn.Dropout2d(0.5), RandomPool2d((2, 2))
        self.extract_core = nn.Sequential(
            nn.Conv2d(1, 24, (9, 5), padding=(4, 2)),  # 80
            selu,
            nn.Conv2d(24, 24, (9, 5), dilation=(1, 3), padding=(4, 2)),  # 72
            selu, do2,
            nn.Conv2d(24, 48, (9, 5), padding=(4, 2)),  # 72
            selu, pool,
            nn.Conv2d(48, 48, (7, 5), padding=(3, 2)),  # 36
            selu, do2,
            nn.Conv2d(48, 48, (7, 5), dilation=(1, 3), padding=(3, 2)),  # 28
            selu,
            nn.Conv2d(48, 48, (7, 5), padding=(3, 2)),  # 28
            selu, pool,  # 14
        )

    @jit.script_method
    def forward(self, inp):
        # type: (List[Tensor]) -> List[Tensor]
        for i in range(len(inp)):
            v = self.extract_core(inp[i].view(1, 1, -1, self.indim))
            inp[i] = v.view(-1, self.tot)
        return inp


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


class localatt(nn.Module):
    def __init__(self, featdim: int, nhid: int, ncell: int, nout: int):
        super(localatt, self).__init__()
        self.convs = CNN(featdim)
        tot = self.convs.tot
        self.rnn1 = tc.nn.LSTM(tot, ncell, 2, batch_first=True, bias=True, bidirectional=True, dropout=0.5)
        for name, param in self.rnn1.named_parameters():
            if 'bias' in name:
                tc.nn.init.uniform_(param, 1 - math.sqrt(1 / ncell), 1 + math.sqrt(1 / ncell))
        self.attn = Attn(ncell, 5)
        self.combine = nn.Linear(ncell * 2, nout)

    def forward(self, ilp: [tc.Tensor]) -> tc.Tensor:
        ilp = self.convs(ilp)
        output = self.rnn1(pack_sequence(ilp))[0]
        del ilp
        output = self.attn(*pad_packed_sequence(output, True))
        return self.combine(output)
