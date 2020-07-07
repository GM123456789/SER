from .LookAhead import Lookahead
from .Normalizer import *
from .RAdam import RAdam
from .Train import train, validate_loop_lazy
from .diffRGrad import diffRGrad
from .lld_dataset import lld_dataset, lld_collate_fn
from .localatt.ISEF import Emotion as localatt
from .validate_lazy import validate_uar, validate_war


def Ranger0(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, k=5, alpha=0.5):
    return Lookahead(RAdam(params, lr, betas, eps, weight_decay), k, alpha)


def Ranger1(params, lr=1e-3, k=5, alpha=0.5):
    return Lookahead(diffRGrad(params, lr), k=k, alpha=alpha)
