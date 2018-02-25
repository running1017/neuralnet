# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse as sp
import itertools as it


class Layer:
    def __init__(self, nodes, lowerLayer):
        self.inNode, self.outNode = nodes
        self.lowerLayer = lowerLayer
        self.inValue = np.zeros(self.inNode)
        self.layernum = self.lowerLayer.layernum + 1

    def __str__(self):
        return '{0}-layer in:{1} out:{2}'.format(self.layernum, self.inNode, self.outNode)


class AffineLayer(Layer):
    def __init__(self, nodes, lowerLayer, optimizer, sigma=0.5):
        super().__init__(nodes, lowerLayer)
        self.weight = np.random.normal(0, sigma, (self.outNode, self.inNode + 1))
        self.grad = np.zeros((self.outNode, self.inNode + 1))
        self.op = optimizer

    def __str__(self):
        return super().__str__() + 'Affine Layer'

    def __call__(self, x):
        self.inValue = self.lowerLayer(x)
        return np.dot(self.weight, np.insert(self.inValue, 0, 1.))

    def backward(self, delta):
        self.lowerLayer.backward(np.dot(delta, self.weight).reshape(-1)[1:])
        self.grad += np.array(np.mat(delta).T * np.mat(np.insert(self.inValue, 0, 1.)))

    def update(self):
        self.weight -= self.op(self.grad)
        self.grad = np.zeros((self.outNode, self.inNode + 1))
        self.lowerLayer.update()


class ActLayer(Layer):
    def __init__(self, node, lowerLayer, actfunc):
        super().__init__((node, node), lowerLayer)
        self.actfunc = actfunc

    def __str__(self):
        return super().__str__() + 'Activation Layer'

    def __call__(self, x):
        self.inValue = self.lowerLayer(x)
        return self.actfunc(self.inValue)

    def backward(self, delta):
        self.lowerLayer.backward(delta*self.actfunc.df(self.inValue))

    def update(self):
        self.lowerLayer.update()


class InputLayer(Layer):
    def __init__(self):
        self.layernum = 0

    def __str__(self):
        return '0-layer Input Layer'

    def __call__(self, x):
        return x

    def backward(self, delta):
        pass

    def update(self):
        pass
