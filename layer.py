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


class ConvLayer(Layer):
    def __init__(self, inputSize, lowerLayer, optimizer, channel, filterSize, filterNum, stride=1, sigma=0.5):
        self.inputSize = inputSize
        self.channel = channel
        self.filterSize = filterSize
        self.filterNum = filterNum
        self.padding = np.floor(np.array(self.filterSize)/2)
        self.stride = stride
        self.outputSize = np.floor((np.array(self.inputSize) - 1) / self.stride) + 1
        super().__init__((np.prod(self.inputSize), np.prod(self.outputSize)), lowerLayer)
        self.paramNum = np.prod(self.filterSize) * self.filterNum + self.filterNum  # filter + bias
        self.hs = np.random.normal(0, sigma, self.paramNum)
        self.grad = np.zeros(self.paramNum)
        self.op = optimizer

    def __str__(self):
        return super().__str__() + 'Convolution Layer'

    def __call__(self, x):
        self.inValue = self.lowerLayer(x)
        return 

    def backward(self):
        pass

    def update(self):
        self.weight -= self.op(self.grad)
        self.lowerLayer.update()

    def multi2single(self, multiIndex, maxIndex):
        base = np.insert(maxIndex, 0, 1)[:-1]
        s = 0
        for i in range(len(base))[::-1]:
            s = (s + multiIndex[i])*base[i]

        return s

    def single2multi(self, singleIndex, maxIndex):
        base = np.insert(maxIndex, 0, 1)[:-1]
        index = singleIndex
        m = []
        q = np.prod(base)
        for i in range(len(base))[::-1]:
            m.insert(int(index/q))
            index %= q
            q /= base[i]

        return m.reverse()

    def makeAdjMatrix(self):  # 2D
        t = sp.lil_matrix((self.outNode, self.inNode, self.paramNum))
        outMaxIndex = np.concatenate([self.outputSize, [self.filterNum]])
        inMaxIndex = np.concatenate([self.inputSize, [self.channel]])
        for outIndex in it.product():
            for filterIndex in it.product():
                t[multi2single(, outMaxIndex)][multi2single(, inMaxIndex)][] = 1

    def convolute(self):
        pass


class PoolingLayer(Layer):
    def __init__(self, nodes, lowerLayer):
        super().__init__(nodes, lowerLayer)

    def __str__(self):
        return super().__str__() + 'Pooling Layer'

    def __call__(self, x):
        pass

    def backward(self):
        pass

    def update(self):
        self.lowerLayer.update()

    def convolute(self):
        pass


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
