# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import neural


def main():
    xs = np.array([[0., 0.],
                  [1., 0.],
                  [0., 1.],
                  [1., 1.]])
    ts = np.array([[0., 1.],
                  [1., 0.],
                  [1., 0.],
                  [0., 1.]])

    const = [{'type': 'Aff', 'size': (2, 4), 'opt': 'Adam'},
             {'type': 'Act', 'func': 'ReLU'},
             {'type': 'Aff', 'size': (4, 2), 'opt': 'Adam'}]

    learner = neural.ConvNeuralNet(const)
    # learner = neural.NeuralNet([2, 10, 10, 2])
    
    n, epoch, batch = 4, 1000, 4

    e = learner.train(xs, ts, n, epoch, batch, log=True)

    for i in range(4):
        print(learner(xs[i]))
    
    plt.plot(e[10:])
    plt.show()

if __name__ == "__main__":
    main()
