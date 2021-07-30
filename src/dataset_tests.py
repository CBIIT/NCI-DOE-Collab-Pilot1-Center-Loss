# reads voxels

import numpy
import sys
import math

import sklearn.utils
from sklearn.preprocessing import StandardScaler

from tf_dataset import *

def chemLabelTest():
    print "starting label test"
    labels = ChemLabels(sys.argv[1], 2)
    loop = False
    first = True
    count = 0

    while not loop:
        batch, loop = labels.get_next_batch()
        print batch
        count += 1
        if count > 3:
            break

def pairTest():
    print "starting pair test"
    featuresFile = GDCDataset(sys.argv[1], 2)
    labelsFile = ChemLabels(sys.argv[2], 2)

    flp = FeatureLabelPair(featuresFile, labelsFile)

    print flp.get_next_batch()
    print flp.get_next_batch()
    print "reset", flp.reset()
    print flp.get_next_batch()

def randomTest():
    featPath = sys.argv[1]
    labPath = sys.argv[2]

    pair = ECFP4Pair(featPath, labPath, batch_size=1024, scaling='standard')

    loop = False
    while not loop:
        x, y, loop = pair.get_next_batch()
        print ".,"

    print "out"

    pair.reset()

    loop = False
    while not loop:
        x, y, loop = pair.get_next_batch()
        print ".,"
    print "out"

def labelTest():
    print "starting label test"
    labels = GDCLabels(sys.argv[1], 1)
    print "numclasses", labels.numClasses
    loop = False
    first = True
    count = 0

    while not loop:
        batch, loop = labels.get_next_batch()
        print batch
        count += 1
        if count > 10:
            break

def looptest():
    print "starting loop test"
    batchSize = 32
    td = GDCDataset(sys.argv[1], batchSize)
    print "feature length", td.numFeatures

    loop = False
    first = True
    count = 0
    while not loop:
        batch, loop = td.get_next_batch()
        count += 1
        if first:
            print batch
            print batch.shape
            first = False
        print ".",

    print ""

    print "num samples", count*batchSize
    td.reset()

def normTest():
    gdcd = GDCDataset(sys.argv[1], 2, scaling='minmax')

    print "num features", gdcd.numFeatures
    print "checking for nans"
    numSamples = 0
    loop = False
    while not loop:
        numSamples += 1
        batch, loop = gdcd.get_next_batch()
        if numpy.isnan(numpy.sum(batch)):
            print "NAN"
    print "Cleared"
    print "num samples", 2*numSamples

def P1B3Test():
    gdc = GDCDataset(sys.argv[1], 1, 'minmax')
    outputFile = open(sys.argv[2], 'w')
    loop = False
    while not loop:
        x, loop = gdc.get_next_batch()
        for l in x:
            l.tofile(outputFile, " ", "%.5f")
            outputFile.write("\n")

    # write these out

def onehotTest():
    labels = sys.argv[1]
    l = ECFP4Random(labels, batch_size = 5)

    y, loop = l.get_next_batch()
    print y
    print y.shape

    print l.all()
    print l.all().shape

def onetimeTest():

    data = ECFP4Pair(sys.argv[1], sys.argv[2], 4)

    count = 1
    stop = False
    while not stop and count < 10:
        fs, ls, stop = data.get_onetime_batch()
        print data.features.currentIndex
        count += 1
        print "fs", fs
        print "ls", ls
        print "--------"

    print "done"

def pairFeatureTest():
    ep = ECFP4Pair([sys.argv[1], sys.argv[2]], sys.argv[3], batch_size=2)

    print ep.get_next_batch()
    print ep.get_next_batch()
    print ep.get_next_batch()
    print ep.get_next_batch()
    print ep.get_next_batch()

    print "reset", ep.reset()

    print ep.get_next_batch()
    print ep.get_next_batch()

    print "randomize", ep.randomize()

    print ep.all()

    print "reset", ep.reset()

    print ep.get_onetime_batch()
    print ep.get_onetime_batch()
    print ep.get_onetime_batch()
    print ep.get_onetime_batch()

def epochTest():
    batch_size = 8
    ep = ECFP4Pair([sys.argv[1]], sys.argv[2], batch_size=batch_size)
    epoch = 4

    total_batch = int(math.ceil(ep.numSamples / batch_size))
    for j in range(epoch):
        print "new epoch"
        for i in range(total_batch):
            print ep.get_next_batch()

        print "reset"
        ep.reset()
        ep.randomize()

def evenClassTest():
    ec = EvenClasses(sys.argv[1], sys.argv[2], 4)

    print ec.get_onetime_batch()
    print "###################"
    print ec.get_onetime_batch()
    print "###################"
    print ec.get_onetime_batch()
    print "###################"
    print ec.get_onetime_batch()
    print "###################"
    print ec.get_onetime_batch()

    ec.reset()
    print "reset"

    print ec.get_next_batch()
    print "###################"
    print ec.get_next_batch()
    print "###################"
    print ec.get_next_batch()
    print "###################"
    print ec.get_next_batch()
    print "###################"
    print ec.get_next_batch()

def stratifiedClassTest():
    sc = StratifiedClasses(sys.argv[1], sys.argv[2], 4)

    print sc.get_next_batch()
    print "#################"

    print sc.get_next_batch()
    print "#################"

    print sc.get_next_batch()

def chunkTest():
    cp = ChunkPair([['dataset_test/1.X', 'dataset_test/2.X', 'dataset_test/3.X'], 
                    ['dataset_test/1.K', 'dataset_test/2.K', 'dataset_test/3.K']], 
#                    ['/p/lscratchh/allen99/hnow/dprep/combine/CTRP.out/CTRP.0._log_con.csv', 
#                    '/p/lscratchh/allen99/hnow/dprep/combine/CTRP.out/CTRP.1._log_con.csv', 
#                    '/p/lscratchh/allen99/hnow/dprep/combine/CTRP.out/CTRP.2._log_con.csv']], 
                    ['dataset_test/1.y', 'dataset_test/2.y', 'dataset_test/3.y'], 
                    4)

    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()
    print cp.get_onetime_batch()



if __name__ == "__main__":
    chunkTest()
#    stratifiedClassTest()
#    evenClassTest()
#    epochTest()
#    pairFeatureTest()

#    onetimeTest()
#    P1B3Test()
#    pairTest()
#    chemLabelTest()
#    normTest()
#    labelTest()
#    looptest()
#    randomTest()
#    onehotTest()