from __future__ import division, print_function, absolute_import

import argparse
import numpy
import os
import math

from tf_dataset import GDCDataset, FeatureLabelPair, ChemLabels
from tf_validationFile import tanimotoCoeff

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainx', default='', action='store', help='path to features')
    parser.add_argument('--trainy', default='', action='store', help='path to features')
    parser.add_argument('--output', default='', action='store', help='path to output')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parseArgs()

    batch_size = 4
    print("args.trainx", args.trainx)
    td = GDCDataset(args.trainx, batch_size, scaling='minmax')
    labels = ChemLabels(args.trainy, batch_size)
    testData = FeatureLabelPair(td, labels)

    print("numSamples", td.numSamples)
    print("numfeatures", td.numFeatures)

    outFile = open(args.output, 'w')

    loop = False
    loopCount = 0
    while not loop:
        X, y, loop = testData.get_next_batch()
        print(X.shape)
        loopCount += 1
        if loop or loopCount > 2:
            break

        for i, x in enumerate(X):
            for j in range(0, i):
                print (i, j)
                taniDistance = tanimotoCoeff(X[i], X[j])
                targetDistance = 1 - math.sqrt((y[i]-y[j])**2)/200
                outFile.write("{:.3},{:.3}\n".format(taniDistance, targetDistance))


    print("num points", loopCount)