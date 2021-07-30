# given pca and some features, test how good reconstruction is

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import argparse
import numpy
import os
import random

from tf_dataset import GDCDataset, FeatureLabelPair
from tf_validationFile import ValidationFile, tanimotoCoeff
from tf_model import getNetwork
from tf_graph import getOp
from tf_encode import Encoder

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='', action='store', help='path to checkpoint folder')
    parser.add_argument('--features', default='', action='store', help='path to features')
    parser.add_argument('--output', default='', action='store', help='root name used to output result csvs and converted features')
    parser.add_argument('--AutoEn_type', default='', action='store', help='which AutoEncoder model to use')
    parser.add_argument('--Optimizer_type', default='', action='store', help='changes cost function type')
    args = parser.parse_args()

    print("the args", dir(args))

    return args

def repleatTest(args):
    print("stupid output", args.output)
    valPath = os.path.join(args.checkpoint_path, args.output+".csv")
    #validationFile = ValidationFile(valPath)
    testCase = open(valPath, 'w')

    print("op type and aut type", args.AutoEn_type, args.Optimizer_type)
    network = getNetwork(args)
    op = getOp(args)

    batch_size = 1
    print("args.features", args.features)
    td = GDCDataset(args.features, batch_size, scaling='minmax')
    testData = FeatureLabelPair(td, labels=None)

    opGraph = op(network(testData.numFeatures, is_training=False))
    encoder = Encoder(args.checkpoint_path, opGraph)

    outputPath = os.path.join(args.checkpoint_path, args.output)
    #outputFile = open(outputPath, 'w')

    loop = False
    loopCount = 0
    lastX = None
    lastEncode = None
    lastDecode = None

    inputSet = set()
    encodedSet = set()
    decodedSet = set()
    xDup = 0
    eDup = 0
    dDup = 0

    numpy.set_printoptions(threshold=1000000000)

    while not loop:
        X, y, loop = testData.get_next_batch()
        x_str = numpy.array_str(X)


        if loopCount == 1:
            print("X as a string")
            print(x_str)

        if loop:
            break

        if loopCount > 100:
            break

        encoded, decoded, cost = encoder.debug_encode(X)

        encoded_str = numpy.array_str(encoded)
        decoded_str = numpy.array_str(decoded)

        if not lastX is None and not lastEncode is None:
            loopCount += 1
            originalDist = tanimotoCoeff(X[0], lastX[0])
            decodeError = tanimotoCoeff(lastDecode, decoded)
            newDist = tanimotoCoeff(lastEncode, encoded)
            #print("shapes", X[0].shape, encoded.shape, decoded.shape)
            #print(originalDist, newDist, decodeError)

            if x_str in inputSet:
                xDup += 1
            else:
                inputSet.update([x_str])

            if encoded_str in encodedSet:
                eDup += 1
            else:
                encodedSet.update([encoded_str])

            if decoded_str in decodedSet:
                dDup += 1
            else:
                decodedSet.update([decoded_str])

            testCase.write("{:.5},{:.5},{:.5},{:.5}\n".format(originalDist, decodeError, 
                            newDist, cost))

        lastX = X
        lastEncode = encoded
        lastDecode = decoded

        #for i in range(0, batch_size):
        #    encoded[i].tofile(outputFile, " ", "%.5f")
        #    outputFile.write("\n")

    print("went through", loopCount, "actual lines")
    print("actually had input", len(inputSet), "and encoded", len(encodedSet), "and decoded", len(decodedSet))
    print("input dups", xDup, "encoded dups", eDup, "decoded dups", dDup)

def encode(args):
    print("stupid output", args.output)
    outPath = os.path.join(args.checkpoint_path, args.output)
    decodePath = outPath.replace('.encode', '.decode')
    avgPath = outPath.replace('.encode', '.avg')
    outputFile = open(outPath, 'w')
    decodeFile = open(decodePath, 'w')
    avgFile = open(avgPath, 'w')

    print("op type and aut type", args.AutoEn_type, args.Optimizer_type)
    network = getNetwork(args)
    op = getOp(args)

    batch_size = 1
    print("args.features", args.features)
    td = GDCDataset(args.features, batch_size, scaling='minmax')
    testData = FeatureLabelPair(td, labels=None)

    opGraph = op(network(testData.numFeatures, is_training=False))
    encoder = Encoder(args.checkpoint_path, opGraph)

    outputPath = os.path.join(args.checkpoint_path, args.output)

    loop = False
    total = 0
    count = 0
    while not loop:
        X, y, loop = testData.get_next_batch()

        if loop:
            break

        count += 1.
        #encoded = encoder.encode(X)
        encoded, decoded, cost = encoder.debug_encode(X)
        total += cost
        encoded.tofile(outputFile, "\t", "%.5f")
        outputFile.write("\n")

        decoded.tofile(decodeFile, "\t", "%.5f")
        decodeFile.write("\n")

    avg = total / count
    avgFile.write(str(avg))

    #validationFile.close()
if __name__ == "__main__":
    args = parseArgs()

    encode(args)