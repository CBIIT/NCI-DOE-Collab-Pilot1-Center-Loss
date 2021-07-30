# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import math
import sys
import argparse
import time

from center_loss import *

def center_penalty_test():
    loop_number = 10
#    samples = np.array([[0, 0, 2], # center [1, 1, 1]
#                        [2, 2, 2],
#                        [0, 2, 2],
#                        [2, 0, 2],
#                        [0, 0, 0],
#                        [2, 2, 0],
#                        [2, 0, 0],
#                        [0, 2, 0],
#                        [0, 0, -2], # center [-1, -1, -1]
#                        [-2, -2, -2],
#                        [0, -2, -2],
#                        [-2, 0, -2],
#                        [0, 0, 0],
#                        [-2, -2, 0],
#                        [-2, 0, 0],
#                        [0, -2, 0]])

    samples = np.array([[0, 0, 4], # center [1, 1, 1]
                        [4, 4, 4],
                        [0, 4, 4],
                        [4, 0, 4],
                        [0, 0, 0],
                        [4, 4, 0],
                        [4, 0, 0],
                        [0, 4, 0],
                        [0, 0, -4], # center [-1, -1, -1]
                        [-4, -4, -4],
                        [0, -4, -4],
                        [-4, 0, -4],
                        [0, 0, 0],
                        [-4, -4, 0],
                        [-4, 0, 0],
                        [0, -4, 0]])

    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1])
    num_classes = 2
    one_hot_labels = np.zeros((len(labels), 2))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    print("num_classes", num_classes)

    with tf.Graph().as_default():
        # test center dist stuff
        X = tf.placeholder(tf.float32, [None, samples.shape[1]], name='ph_input_tensor')
        y_true = tf.placeholder(tf.int32, [None], name='ph_y_true')
        y_true_oh = tf.placeholder(tf.int32, [None, num_classes], name='ph_y_true_oh')

        loss, centers, centers_update_op = get_center_loss(X, y_true, 1, num_classes)
        center_distance = calc_center_distances(X, y_true_oh)

        # Initializing the variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        for i in range(loop_number):
            feed_dict = {X : samples, y_true : labels, y_true_oh : one_hot_labels}
            calc_loss, calc_centers, calc_centers_update, calc_dist = sess.run([loss, centers, centers_update_op, 
                                                                center_distance], feed_dict)

            print("loss", calc_loss) # should converge to 3/sqrt(32)
            print("centers", calc_centers) # should converge to [1, 1, 1] and [-1, -1, -1] /sqrt(32)
            print("calc_dist", calc_dist) # should converge to 6/sqrt(32)
            print("--------------------------------------")

if __name__ == "__main__":
    center_penalty_test()

