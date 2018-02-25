# -*- coding: utf-8 -*-

import numpy as np
import sys
import time
import layer
import optimizer as op
import actfunc as ac


class NeuralNet:
    def __init__(self, nodes, classification=False):   # [M,H1,H2,...,N]
        self.nodes = nodes
        lastLayer = layer.InputLayer()
        for l in range(len(nodes) - 1):
            opt = op.Adam((nodes[l], nodes[l + 1]))
            lastLayer = layer.AffineLayer((nodes[l], nodes[l + 1]), lastLayer, opt)
            if l < len(nodes) - 2:
                act = ac.ReLU()
            elif classification:
                act = ac.Softmax()
            else:
                act = ac.Identity()
            lastLayer = layer.ActLayer(nodes[l + 1], lastLayer, act)

        self.outputLayer = lastLayer
    
    def __call__(self, x):
        return self.outputLayer(x)

    def train(self, xs, ts, n, epoch, batch, log=False):
        error = []
        start = time.time()

        for e in range(epoch):
            sffidx = np.random.permutation(n)
            for s in range(0, n, batch):
                for j in range(s, min(s + batch, n)):
                    ys = self(xs[sffidx[j]])
                    self.outputLayer.backward(ys - ts[sffidx[j]])
                self.outputLayer.update()
            if log:
                error.append(sum([np.sum((self(xs[i]) - ts[i])**2) for i in range(n)]))

            t = time.time() - start
            prog = float(e+1)/epoch
            ltime = t*(1/prog - 1)
            hour = int(ltime/3600)
            ltime %= 3600
            minute = int(ltime/60)
            ltime %= 60
            second = int(ltime)
            sys.stdout.write('\r' + '{0}%    ---{1}h{2}m{3}s lefts---'.format(int(prog*100), int(hour), int(minute), int(second)))

        sys.stdout.write('\n')
        return error


class ConvNeuralNet(NeuralNet):
    def __init__(self, const):
        self.const = const
        lastLayer = layer.InputLayer()
        for l in range(len(const)):
            layerInfo = const[l]
            if layerInfo['type'] == 'Aff':
                if layerInfo['opt'] == 'Adam':
                    opt = op.Adam(layerInfo['size'])
                elif layerInfo['opt'] == 'SGD':
                    opt = op.SGD()

                lastLayer = layer.AffineLayer(layerInfo['size'], lastLayer, opt)
            
            elif layerInfo['type'] == 'Act':
                if layerInfo['func'] == 'ReLU':
                    act = ac.ReLU()
                elif layerInfo['func'] == 'Sigmoid':
                    act = ac.Sigmoid()
                elif layerInfo['func'] == 'Softmax':
                    act = ac.Softmax()

                lastLayer = layer.ActLayer(lastLayer.outNode, lastLayer, act)

        self.outputLayer = lastLayer
