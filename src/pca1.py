#!/usr/bin/env python2.7

###################################################################################
## compute principal components, and save model to a file
##
## more documentation to come
###################################################################################

import argparse
import errno
import os
import sys

import numpy as np
import numpy.random

from sklearn.externals import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_dataset(xfname,lfile):
    delim='\t'
    sys.stderr.write("Reading data from {}\n".format(xfname))
    X_label=[]
    if lfile != '' :
       fm_fh = open(lfile)
       tx=[]
       for val in fm_fh :
         val=val.rstrip()
         vals=val.split('\t')
         tx.append((int(val[0]),vals[1]))

       X_label = sorted(tx, key=lambda f:f[0], reverse=False)

    f = open(xfname)
    line = f.readline().rstrip()
    cols = line.split(delim)
    print "num_cols",len(cols),delim
    X = np.genfromtxt(xfname, delimiter=delim)
    return X, X_label

def sprint_features(top_features, num_features=20):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= num_features: return
        #print "feature debug",feature,feature[0],feature[1]
        #str += '{}\t{:.5f}\n'.format(feature[1], feature[0])
        str += '{}\t{:.5f}\n'.format(feature[1], feature[0])
    return str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', action='store', help='store results files to a specified directory')
    parser.add_argument('--num_comp', default=100, action='store', help='number of principal')
    parser.add_argument('-l', '--label', default=1, action='store', help='convert classifier labels (y) into a binary vector')
    parser.add_argument('-j', '--jobid', default=1, action='store', help='numeric identifier to track model')
    parser.add_argument('--featl', default='', help='feature label file')
    parser.add_argument('--trainx', default='', help='training file labels')
    parser.add_argument('--save', default='', help='file output for model file')
    args = parser.parse_args()

    print "huh",args.num_comp
    X, labels = read_dataset(args.trainx, args.featl )
    pca=PCA() 
    pca.fit(X,args.num_comp)
    lval=len( pca.explained_variance_ )
    print pca.explained_variance_
    joblib.dump(pca, args.save)

if __name__ == '__main__':
    main()
