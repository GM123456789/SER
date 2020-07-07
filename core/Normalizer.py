import math

import numpy as np


class Normalizer:
    def __init__(self, **params):
        self.params: dict = params

    def fit(self, data: [np.ndarray]) -> dict:
        pass

    def transform(self, data: [np.ndarray]):
        pass

    def __call__(self, data):
        if len(self.params) == 0:
            self.fit(data)
        self.transform(data)

    def __str__(self):
        return str(self.params)


class GlobalRangeNorm(Normalizer):
    def fit(self, data: list) -> dict:
        self.params['mean'] = sum(i.sum() for i in data) / sum(i.size for i in data)
        self.params['max'], self.params['min'] = max(i.max() for i in data), min(i.min() for i in data)
        return self.params

    def transform(self, data: [np.ndarray]):
        r = self.params['max'] - self.params['min']
        for i in range(len(data)):
            data[i] -= self.params['mean']
            data[i] /= r


class GlobalGaussNorm(Normalizer):
    def fit(self, data: list) -> dict:
        size = sum(i.size for i in data)
        self.params['mean'] = mean = sum(i.sum() for i in data) / size
        self.params['std'] = math.sqrt(sum(((i - mean) ** 2).sum() for i in data) / size)
        return self.params

    def transform(self, data: list):
        for i in range(len(data)):
            data[i] -= self.params['mean']
            data[i] /= self.params['std']


class LocalGaussStd(Normalizer):
    def __init__(self, **params):
        if len(params) == 0:
            params['eps'] = 1e-8
        super().__init__(**params)

    def transform(self, data: [np.ndarray]):
        for i in data:
            i -= i.mean(0)
            i /= i.std(0).clip(min=self.params['eps'])


class LocalGaussNorm(Normalizer):
    def __init__(self, **params):
        if len(params) == 0:
            params['eps'] = 1e-8
        super().__init__(**params)

    def transform(self, data: [np.ndarray]):
        for i in data:
            i -= i.mean()
            i /= i.std().clip(min=self.params['eps'])


class GlobalMixedStd(Normalizer):
    def fit(self, data: [np.ndarray]) -> dict:
        llds, vggs = zip(*((i[:, :32], i[:, 32:]) for i in data))
        vgg_size, lld_size = sum(i.size for i in vggs), sum(i.size for i in llds)

        self.params['vgg_mean'] = mean = sum(i.sum() for i in vggs) / vgg_size
        self.params['vgg_std'] = math.sqrt(sum(np.square(i - mean).sum() for i in vggs) / vgg_size)
        self.params['lld_mean'] = mean = sum(i.sum() for i in llds) / lld_size
        self.params['lld_std'] = math.sqrt(sum(np.square(i - mean).sum() for i in llds) / lld_size)
        return self.params

    def transform(self, data: [np.ndarray]):
        for i in data:
            lld, vgg = i[:, :32], i[:, 32:]
            lld -= self.params['lld_mean']
            lld /= self.params['lld_std']
            vgg -= self.params['vgg_mean']
            vgg /= self.params['vgg_std']
