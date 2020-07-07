import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as tcf


class _normal_2d(nn.Module):
    def __init__(self, kernal_size=2, eps=0):
        super().__init__()
        self.ksize = tuple(kernal_size) if type(kernal_size) == int else kernal_size
        self.p = self.ksize[0] * self.ksize[1]
        self.actv = nn.Softplus(threshold=10)
        self.eps = eps

    def weightedAverage(self, inp: torch.Tensor):
        bsz, chn, h, w = inp.size()
        inp = tcf.unfold(inp, self.ksize, stride=self.ksize)
        inp = inp.view(bsz, chn, self.p, -1).transpose_(2, 3).flatten(0, 2)
        weight = self.actv(inp)
        weight /= weight.sum(1, keepdim=True)
        inp = (inp * weight).sum(1, keepdim=True)
        inp = inp.view(bsz, chn, -1, 1).transpose_(2, 3)
        return inp.view(bsz, chn, h // self.ksize[0], w // self.ksize[1])

    def weightedMaxPool(self, inp: torch.Tensor):
        bsz, chn, h, w = inp.size()
        inp = tcf.unfold(inp, self.ksize, stride=self.ksize)
        inp = inp.view(bsz, chn, self.p, -1).transpose_(2, 3).flatten(0, 2)
        inp = inp.gather(1, torch.multinomial(self.actv(inp.data), 1))
        inp = inp.view(bsz, chn, -1, 1).transpose_(2, 3)
        return inp.view(bsz, chn, h // self.ksize[0], w // self.ksize[1])

    def forward(self, inp: torch.Tensor):
        if self.training:
            return self.weightedMaxPool(inp)
        else:
            return self.weightedAverage(inp)


class _jit_2d(jit.ScriptModule):
    __constants__ = ['ksize', 'p']

    def __init__(self, kernal_size=2):
        super().__init__()
        self.ksize = tuple(kernal_size) if type(kernal_size) == int else kernal_size
        self.p = self.ksize[0] * self.ksize[1]
        self.actv = nn.Softplus(threshold=10)

    @jit.script_method
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        bsz, chn, h, w = inp.shape
        inp = tcf.unfold(inp, self.ksize, stride=self.ksize)
        inp = inp.view(bsz, chn, self.p, -1).transpose_(2, 3).flatten(0, 2)
        if self.training:
            inp = inp.gather(1, torch.multinomial(self.actv(inp.data), 1))
        else:
            weight = self.actv(inp)
            weight /= weight.sum(1, keepdim=True)
            inp = (inp * weight).sum(1, keepdim=True)
        inp = inp.view(bsz, chn, -1, 1).transpose_(2, 3)
        return inp.view(bsz, chn, h // self.ksize[0], w // self.ksize[1])


def RandomPool2d(ksize, ts=False):
    if ts:
        return _jit_2d(ksize)
    return _normal_2d(ksize)


if __name__ == '__main__':
    a = RandomPool2d((1, 2,), ts=True)
    a.train()
    # a.eval()
    b = torch.tensor([[[[1, 2], [0.5, 0.51], [1, 20], [0.5, 0.51]]]])
    # b=torch.zeros(2,1,4,4)
    print(b)
    print(a(b))
