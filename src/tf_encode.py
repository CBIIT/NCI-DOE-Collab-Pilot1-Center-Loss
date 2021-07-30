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

import tf_main
from tf_graph import getOp
import os

def encodeFeatures(args):
    chunks = tf_main.buildSplitDatasets(args)

    with tf.Graph().as_default():
        checkpoint = os.path.join(args.outputFolder, args.out_check_name)
        opGraph = getOp(args)(args, chunks[0], checkpoint=checkpoint)

        for chunk in chunks:
            current_label_name = chunk.current_label_name()
            batches = []
            stop = False
            while not stop:
                data = chunk.get_onetime_batch()
                stop = data[-1]
                labels = data[-2]

                encoded = opGraph.encode(data)

                batches.append(encoded)

            # make the filename for the dump
            dump_filename = os.path.basename(current_label_name)
            dump_filename = os.path.splitext(dump_filename)[0]+".encoded"
            dump_filename = os.path.join(args.outputFolder, dump_filename)

            # dump
            print("dumping to ", dump_filename)
            concated = np.concatenate(batches)
            np.save(dump_filename, concated)
