from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.preprocessing import StandardScaler

import numpy as np

class MNIST:
    def __init__(self, batch_size):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.numSamples = self.mnist.train.num_examples
        self.batch_size = batch_size
        self.numFeatures = 28*28

        self.index = 0
        self.numClasses = 10

    def mnist_get(self):
        return self.mnist.train.next_batch(self.batch_size)

    def get_next_batch(self):
        batch_xs, batch_ys = self.mnist_get()
        batch_ys = batch_ys.astype(int)

        # the mnist data object already resets and loops when calling next_batch
        self.index += self.batch_size
        loop = False
        if self.index >= self.numSamples:
            loop = True
            self.index = self.index-self.numSamples

        return batch_xs, batch_ys, loop

    def get_onetime_batch(self):
        batch_xs, batch_ys = self.mnist_get()
        batch_ys = batch_ys.astype(int)

        # the mnist data object already resets and loops when calling next_batch
        self.index += self.batch_size
        loop = False
        if self.index >= self.numSamples:
            loop = True
            if self.index > self.numSamples:
                batch_xs = batch_xs[:-(self.index-self.numSamples)]
                batch_ys = batch_xs[:-(self.index-self.numSamples)]
            self.reset()
            self.index = 0

        return batch_xs, batch_ys, loop

    def reset(self):
        # there isn't really a way to reset this data object so I just open a new one
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def randomize(self):
        pass

    def all(self):
        return self.mnist.train.next_batch(self.numSamples)

class MNIST_val(MNIST):
    def __init__(self, batch_size):
        MNIST.__init__(self, batch_size)
        self.numSamples = self.mnist.validation.num_examples

    def mnist_get(self):
        return self.mnist.validation.next_batch(self.batch_size)

    def fullLabels(self):
        return self.mnist.validation.labels

    def all(self):
        return self.mnist.validation.next_batch(self.numSamples)

if __name__ == "__main__":
    print("running StandardScalar test")
    mnist = MNIST(512)

    for i in range(1):
        x, y, loop = mnist.get_next_batch()

        avgMean = x.mean(0).mean()
        print("average mean is", avgMean)

        avgVar = x.var(0).mean()
        print("average var is", x.var(0))
