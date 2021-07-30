# testDA_run.py

import os
import sys

import timeit
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from denoisingAutoencoder import *


#ofile = open("METABRIC_DAparamTrained.txt",'w')
ofname=sys.argv[2]
ofile = open(ofname,'w')

W = []
b = []

learning_rate_all = [0.005]
batch_size_all = [1]
training_epochs_all = [500]

nHidden=100
corruptionLevel=0.1

#==============================================================
# Dataset format: 
# Rows are samples
# Columns are attributes/features
# Note: METABRIC dataset is a transpose of the required format
#==============================================================
#dataset='./data/METABRIC_dataset_trans.dat'
#dataset='/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite/METABRIC_dataset_trans.dat'
dataset=sys.argv[1]


for iter1 in range(len(learning_rate_all)):
    learning_rate = learning_rate_all[iter1]

    for iter2 in range(len(batch_size_all)):
        batch_size = batch_size_all[iter2]

        for iter3 in range(len(training_epochs_all)):
            training_epochs = training_epochs_all[iter3]
            W,b,cost = test_DA(learning_rate,training_epochs,dataset,batch_size, nHidden, corruptionLevel)

            ofile.write(str(learning_rate)+ '\t' + str(batch_size) + '\t' + str(training_epochs) + '\t' + str(cost) + '\n')

ofile.close()
