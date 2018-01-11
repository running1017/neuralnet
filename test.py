# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import neural


def main():
    xs = np.array([[0., 0.],
                  [1., 0.],
                  [0., 1.],
                  [1., 1.]])
    ys = np.array([[0., 1.],
                  [1., 0.],
                  [1., 0.],
                  [0., 1.]])

    learner = neural.NeuralNet([2, 3, 4, 2], classification=False)

    N, epoch, batch = 4, 1000, 4
    e = learner.train(xs, ys, N, epoch, batch, 1, (0.9, 0.999, 0.01, 1e-8), log=True)

    for i in range(4):
        print(learner.forward(xs[i]))

    plt.clf()
    plt.plot(e)
    plt.show()


if __name__ == "__main__":
    main()
