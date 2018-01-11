# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image as im
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import gzip
import neural

def main():
    file_dir = 'mnist/'
    file_list = {
        'train_img': file_dir + 'train-images-idx3-ubyte.gz',
        'train_label': file_dir + 'train-labels-idx1-ubyte.gz',
        'test_img': file_dir + 't10k-images-idx3-ubyte.gz',
        'test_label': file_dir + 't10k-labels-idx1-ubyte.gz'
    }
    
    def load_mnist(filename, size, ofs):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=ofs)
        return data.reshape(-1, size)
    
    tr_data = load_mnist(file_list['train_img'], 784, 16)
    tr_label = LabelBinarizer().fit_transform(load_mnist(file_list['train_label'], 1, 8))
    te_data = load_mnist(file_list['test_img'], 784, 16)
    te_label = LabelBinarizer().fit_transform(load_mnist(file_list['test_label'], 1, 8))
    
    N = 10000
    epoch = 10
    batch = 100
    const = [('Aff', ((784, 1000), 'Adam')),
             ('Act', 'ReLU'),
             ('Aff', ((1000, 1000), 'Adam')),
             ('Act', 'ReLU'),
             ('Aff', ((1000, 10), 'Adam')),
             ('Act', 'Softmax')]
    n = neural.ConvNeuralNet(const)

    print('N={0}, epoch={1}, batch={2}\nneural={3}'.format(N, epoch, batch, n.const))
    
    e = n.train(tr_data[:N], tr_label[:N], N, epoch, batch, log = True)
    
    accuracy = 0.
    testN = 10000
    for i in range(testN):
        y = n(te_data[i])
        if i%100 == 0:
            print('t={0} y={1}'.format(te_label[i].argmax(), y.argmax()))
        accuracy += te_label[i][np.argmax(y)]

    print(accuracy/testN)

    plt.plot(e)
    plt.show()

if __name__ == '__main__': main()