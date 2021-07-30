#!/usr/bin/env python2.7

###################################################################################
## simple template for running a basic classifer initially on gene expression data
##
## more documentation to come
###################################################################################

import argparse
import errno
#import h5py
import os
import sys

import numpy as np
import numpy.random

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve, r2_score, mean_squared_error
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.utils

from sklearn import preprocessing

numpy.set_printoptions(threshold=numpy.nan)

def read_dataset(xfname,yfname,ymapname,label,rcols,isRegression):
    sys.stderr.write("Reading data from {}\n".format(xfname))
    X_label=[]
    if ymapname != '' :
      fm_fh = open(ymapname)
      tx=[]
      for val in fm_fh :
         val=val.rstrip()
         vals=val.split('\t')
         tx.append((int(val[0]),vals[1]))

      X_label = sorted(tx, key=lambda f:f[0], reverse=False)

    # read in feature vectors
    f = open(xfname)
    line = f.readline().rstrip()
    cols = line.split()

    ## I had to explecitly check for float and int typtes, this should be backword compatible but to be 
    ## safe for now until fully tested, I'm just reading the GDC data with the old format
    if isRegression :
      if rcols != [] :
        X = np.genfromtxt(xfname, usecols=rcols, dtype=('float','int'), unpack=True)
      else :
        X = np.genfromtxt(xfname, dtype=('float','int'), unpack=True)
      X=np.transpose(X)
      y = np.genfromtxt(yfname, dtype=('float','int'))
    else :
      if rcols != [] :
        X = np.loadtxt(xfname, usecols=rcols)
      else :
        X = np.loadtxt(xfname)
      y = np.loadtxt(yfname)


    if label != -1 :
      y = np.transpose(map(lambda x: 1 if x == label else 0, y))
      y = np.transpose(y)
    print X.shape,y.shape
    return X, y, X_label

def reLU(xval) :
    return max(0,xval)

def sngl_layer_encode(Xupdate2,weight,bval) :
    num_ex = Xupdate2.shape[0]
    num_edges = weight.shape[0]
    num_nodes = weight.shape[1]
    num_fe = Xupdate2.shape[1]
    assert num_fe == num_edges
    output=np.zeros( (num_ex, num_nodes) )
    print "check",num_ex,num_nodes,num_edges
    for ex_i in range(num_ex) :
        for node_i in range(num_nodes) :
            nsum=0
            for edge_i in range(num_edges) :
                a = weight[edge_i][node_i]
                x = Xupdate2[ex_i][edge_i]
                nsum += (a*x)
            nsum += bval[node_i]
            output[ex_i][node_i] = reLU(nsum)
    return output

def encode_features(flst,Xorig):
    delim=' '
    sys.stderr.write("Reading data from {}\n".format(flst))
    fm_fh = open(flst)
    tx=[]
    for file in fm_fh :
      file=file.rstrip()
      f = open(file)
      line = f.readline().rstrip()
      cols = len(line.split(delim))
      print "read ",file
      weights = np.genfromtxt(file, delimiter=delim, usecols=range(0,cols-1))
      bvals = np.genfromtxt(file, delimiter=delim, usecols=[cols-1])
      tx.append((np.transpose(weights),np.transpose(bvals)))

    Xupdate=Xorig
    cnt=0
    vfunc = np.vectorize(reLU)
    for weight,bval in tx :
       print "debugI",cnt,Xupdate.shape,weight.shape,bval.shape
       res=np.dot(Xupdate,weight)
       res += bval
       res = np.apply_along_axis(vfunc,1,res)
       print "debugO",cnt,res.shape
       Xupdate = res
       #Xupdate2=sngl_layer_encode(Xupdate2,weight,bval)
       #for i in range(Xupdate.shape[0]) :
         #for j in range(Xupdate.shape[1]) :
            #assert Xupdate[i][j] == Xupdate2[i][j]
       #assert Xupdate == Xupdate2
       cnt+=1
       
    print "final encodings",Xupdate
    return Xupdate

def tf_encode_features(fileroot, X):
    normed = preprocessing.MinMaxScaler().fit_transform(X)
    en = Encoder(fileroot, X.shape[1], AutoEncoder)

    return en.encode(normed)

def score_format(metric, score, eol='\n'):
    return '{:<15} = {:.5f}'.format(metric, score) + eol

def top_important_features(clf, feature_names, num_features):
    if hasattr(clf, "booster"):
        fi = clf.booster().get_fscore()
        return None
    elif not hasattr(clf, "feature_importances_"):
        return
    else:
        fi = clf.feature_importances_
        features = [ (f, n) for f, n in zip(fi, feature_names)]
        #top = sorted(features, key=lambda f:f[0], reverse=True)[:num_features]
        top = sorted(features, key=lambda f:f[0], reverse=True)

    return top

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
    parser.add_argument('-f', '--folds', default=3, action='store', help='number of folds for cross validation if test data is not provided')
    parser.add_argument('-o', '--outdir', action='store', help='store results files to a specified directory')
    parser.add_argument('-p', '--prefix', action='store', help='output prefix')
    parser.add_argument('-s', '--skipcols', default=1, action='store', help='number of columns before the y column')
    parser.add_argument('-m', '--multi', default=False, action='store_true', help='ignores --label flag and performs multiclass classification')
    parser.add_argument('-l', '--label', default=1, action='store', help='convert classifier labels (y) into a binary vector')
    parser.add_argument('-j', '--jobid', default='1', action='store', help='numeric identifier to track model')
    parser.add_argument('--ae_encoder', default=False, action='store', help='file with list of weight matrices')
    parser.add_argument('--tf_encoder', default=False, action='store', help='path to tensorflow folder structure')
    parser.add_argument('--pca', default=False, action='store', help='file with list of weight matrices')
    parser.add_argument('--save_model', default=False, action='store', help='write models to file')
    parser.add_argument('--trainy', default='', help='training file output values')
    parser.add_argument('--trainx', default='', help='training file features ')
    parser.add_argument('--trainm', default='', help='training file labels mapped to names')
    parser.add_argument('--rand_feat', default='', help='number of random features to pick')
    parser.add_argument('--regression', action='store_true', help='regresion instead of classification')
    parser.add_argument('test', default='', nargs='?', help='testing file')
    args = parser.parse_args()

    pid=args.jobid
    iarr=[]
    if args.rand_feat :
        iarr = np.arange( int(args.rand_feat) )
        np.random.shuffle( iarr )
        pid += ".r" + args.rand_feat
    else :
        pid += ".rNone"

    X, y, labels = read_dataset(args.trainx, args.trainy,args.trainm, int(args.label),iarr,args.regression)
    if args.ae_encoder :
         ## doesn't work with random feature selection
        assert not args.rand_feat
        X = encode_features(args.ae_encoder,X)
    elif args.pca :
        pca=joblib.load(args.pca)
        print "orig dim",X.shape
        X = pca.transform(X)
        print "pca dim",X.shape
        #print(X)

    # sys.exit(0)
    if args.outdir:
        prefix=args.outdir
    else :
        prefix="./"
    classifiers = [
                    #('XGBoost', XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)),
                    ('DC1',  DummyClassifier(strategy='uniform')),
                    ('DC2',  DummyClassifier(strategy='most_frequent')),
                    ('RF',  RandomForestClassifier(n_estimators=100, n_jobs=10)),
                    ('SVM', SVC()),
                    ('LogRegL1', linear_model.LogisticRegression(penalty='l1')),
                    ('Ada', AdaBoostClassifier(n_estimators=100)),
                    ('KNN', KNeighborsClassifier()),
                  ]

    rmodel={ 'LogRegL1' : True }


    for name, clf in classifiers:
        if args.regression :
            assert rmodel.has_key(name)
        sys.stderr.write("\n> {}\n".format(name))

        train_scores, test_scores = [], []
        probas = None
        tests = None
        preds = None
        roc_auc_score = None

        print "name of class", name
        skf = StratifiedKFold(y, n_folds=int(args.folds), shuffle=True)
        for i, (train_index, test_index) in enumerate(skf):
            # create training and test folds
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print "huh",i,X_train.shape,y_train.shape
            clf.fit(X_train, y_train)

            # get scores
            train_scores.append(clf.score(X_train, y_train))
            test_scores.append(clf.score(X_test, y_test))
            #sys.stderr.write("  fold #{}: score={:.3f}\n".format(i, clf.score(X_test, y_test)))

            # get predictions
            y_pred = clf.predict(X_test)
            # pred is predictions tests is ground truth
            preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
            tests = np.concatenate((tests, y_test)) if tests is not None else y_test

            #if hasattr(clf, "predict_proba"):
                #probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
                #probas = np.concatenate((probas, probas_)) if probas is not None else probas_
            pred_fname = "{}{}.{}.{}.pred".format(prefix, pid, name,i)
            with open(pred_fname, "w") as pred_file:
               for it in range(len(y_pred)) :   
                  outstr = str(y_pred[it]) + " " + str(y_test[it])
                  pred_file.write( outstr + "\n")

        #top_features = top_important_features(clf, labels,len(labels))
        top_features = None
        if top_features is not None:
            fea_fname = "{}.{}.features".format(prefix, name)
            with open(fea_fname, "w") as fea_file:
                fea_file.write(sprint_features(top_features,len(top_features)))

        roc_auc_score = None
        if probas is not None and not args.multi:
            fpr, tpr, thresholds = roc_curve(tests, probas[:, 1])
            #fpr, tpr, thresholds = roc_curve(tests, preds)
            roc_auc_score = auc(fpr, tpr)
            roc_fname = "{}.{}.ROC".format(prefix, name)
            with open(roc_fname, "w") as roc_file:
                roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR'])+'\n')
                for ent in zip(thresholds, fpr, tpr):
                    roc_file.write('\t'.join("{0:.5f}".format(x) for x in list(ent))+'\n')

        if args.save_model :
           model_fname = "{}{}.{}.model".format(prefix, pid, name)
           joblib.dump(clf,model_fname)
        scores_fname = "{}{}.{}.scores".format(prefix, pid, name)

        print "write scores to",scores_fname
        if args.regression :
           ms = 'r2_score mean_squared_error'.split()
        else :
           ms = 'accuracy_score f1_score precision_score recall_score log_loss'.split()
        with open(scores_fname, "w") as scores_file:
            for m in ms:
                if m == 'f1_score' or m == 'precision_score' or m == 'recall_Score':
                    s = getattr(metrics, m)(tests, preds, average='macro')
                else:
                    s = getattr(metrics, m)(tests, preds)

                scores_file.write(score_format(m, s))

            avg_train_score = np.mean(train_scores)
            avg_test_score = np.mean(test_scores)

            if roc_auc_score is not None:
                scores_file.write(score_format('roc_auc_score', roc_auc_score))

            scores_file.write(score_format('avg_test_score', avg_test_score))
            scores_file.write(score_format('avg_train_score', avg_train_score))
            scores_file.write('\nModel:\n{}\n\n'.format(clf))

        sys.stderr.write('  test={:.5f} train={:.5f}\n'.format(avg_test_score, avg_train_score))

    sys.stderr.write("\n")

if __name__ == '__main__':
    main()
