# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as ss


class Sigmoid:
    def __call__(self, x):
        return ss.expit(x)

    def df(self, x):
        return ss.expit(x) * (1. - ss.expit(x))


class ReLU:
    def __call__(self, x):
        return np.maximum(x, 0)

    def df(self, x):
        return np.float64(x>=0)


class Softmax:
    def __call__(self, x):
        c = np.max(x)
        expX = np.exp(x - c)
        return expX/np.sum(expX)
        
    def df(self, x):
        return 1.


class Identity:
    def __call__(self, x):
        return x

    def df(self, x):
        return 1.
