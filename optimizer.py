# -*- coding: utf-8 -*-

import numpy as np


class SGD:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.t = 1

    def __call__(self, grad):
        return eta*grad


class Adam:
    def __init__(self, size, beta1=0.9, beta2=0.999, eta=0.01, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.epsilon = epsilon
        self.m = np.zeros((size[1], size[0] + 1))
        self.v = np.zeros((size[1], size[0] + 1))
        self.t = 1

    def __call__(self, grad):
        self.m = self.beta1*self.m + (1.-self.beta1)*grad
        self.v = self.beta2*self.v + (1.-self.beta2)*np.power(grad, 2)
        mHat = self.m/(1.-self.beta1**self.t)
        vHat = self.v/(1.-self.beta2**self.t)
        self.t += 1
        return self.eta/(np.sqrt(vHat) + self.epsilon)*mHat
