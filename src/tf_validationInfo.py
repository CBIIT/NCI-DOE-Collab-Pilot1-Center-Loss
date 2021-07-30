from __future__ import division, print_function, absolute_import

import numpy as np
import os
import math

from sklearn.metrics import accuracy_score, f1_score, \
                            precision_score, recall_score, \
                            r2_score, log_loss, mean_squared_error

class RSquaredResults:
    def __init__(self, Y=None, P=None, cost=None):
        self.Y = Y
        self.P = P

        # sometimes this object holds several sets of predictions
        if self.P.shape != self.Y.shape:
            # in that case, take the average r2_score across all sets of preds
            self.rsquared = np.mean([r2_score(self.Y, self.P[:,i]) for i in range(self.P.shape[1])])
        else:
            self.rsquared = r2_score(self.Y, self.P)

        self.cost = cost

    # given several val_infos, combines all data into one val_info
    @staticmethod
    def joinValInfos(all_val_info):
        allY = np.concatenate([ vi.Y for vi in all_val_info ])
        allP = np.concatenate([ vi.P for vi in all_val_info ])

        avg_cost = np.mean([vi.cost for vi in all_val_info])

        return RSquaredResults(Y=allY, P=allP, cost=avg_cost)

    # given one additional val_info, concatenates it along axis=1.
    # used for running several dropout masks for the same dataset
    def concatPredictions(self, other_val_info):
        self.P = np.concatenate([self.P, other_val_info.P], axis=1)

    def compare(self, other):
        if self.cost < other.cost:
            print("new best score", self.cost)
            return 1
        elif self.cost == other.cost:
            return 0
        else:
            return -1

    def truth_pred(self):
        # save predictions to a file
        if self.P is None:
            return self.Y
        else:
            return np.column_stack((self.Y, self.P))

    def printStatistics(self):
        print("r2_score:", self.rsquared)

    def to_log_string(self):
        return ','.join(map(str, [self.cost, self.rsquared]))

class ClassificationAutoencoderResults:
    def __init__(self, X, y, pred_X, pred_y, cost):
        self.X = X
        self.y = y
        self.pred_X = pred_X
        self.pred_y = pred_y
        self.cost = cost

    # given several val_infos, combines all data into one val_info
    @staticmethod
    def joinValInfos(all_val_info):
        allX = np.concatenate([ vi.X for vi in all_val_info ])
        ally = np.concatenate([ vi.y for vi in all_val_info ])
        all_pred_X = np.concatenate([ vi.pred_X for vi in all_val_info ])
        all_pred_y = np.concatenate([ vi.pred_y for vi in all_val_info ])

        avg_cost = np.mean([vi.cost for vi in all_val_info])

        return ClassificationAutoencoderResults(X=allX, y=ally, 
            pred_X=all_pred_X, pred_y=all_pred_y, cost=avg_cost)

    def compare(self, other):
        if self.cost < other.cost:
            print("new best score", self.cost)
            return 1
        elif self.cost == other.cost:
            return 0
        else:
            return -1

    def printStatistics(self):
        print("cost %.3f" % self.cost)
        print("reconstruction err %.3f:" % self.recon_error())
        print("accuracy %.3f precision %.3f recall %.3f" % 
            (accuracy_score(self.y, self.pred_y), \
            precision_score(self.y, self.pred_y, average='weighted'), \
            recall_score(self.y, self.pred_y, average='weighted')))

    def recon_error(self):
        # rmse
        return math.sqrt(mean_squared_error(self.X, self.pred_X))

    def f1(self):
        return f1_score(self.y, self.pred_y, average='weighted')

    def truth_pred(self):
        return np.column_stack((self.y, self.pred_y))

class CenterLossAEResults(ClassificationAutoencoderResults):
    def __init__(self, X, y, pred_X, pred_y, cost, center_loss):
        self.X = X
        self.y = y
        self.pred_X = pred_X
        self.pred_y = pred_y
        self.cost = cost
        self.center_loss = center_loss

    # given several val_infos, combines all data into one val_info
    @staticmethod
    def joinValInfos(all_val_info):
        allX = np.concatenate([ vi.X for vi in all_val_info ])
        ally = np.concatenate([ vi.y for vi in all_val_info ])
        all_pred_X = np.concatenate([ vi.pred_X for vi in all_val_info ])
        all_pred_y = np.concatenate([ vi.pred_y for vi in all_val_info ])

        avg_cost = np.mean([vi.cost for vi in all_val_info])
        avg_cl = np.mean([vi.center_loss for vi in all_val_info])

        return CenterLossAEResults(X=allX, y=ally, 
            pred_X=all_pred_X, pred_y=all_pred_y, cost=avg_cost, center_loss=avg_cl)

    def printStatistics(self):
        print("cost %.3f" % self.cost)
        print("reconstruction err %.3f:" % self.recon_error())
        print("accuracy %.3f precision %.3f recall %.3f" % 
            (accuracy_score(self.y, self.pred_y), \
            precision_score(self.y, self.pred_y, average='weighted'), \
            recall_score(self.y, self.pred_y, average='weighted')))
        print("center_loss %.3f" % self.center_loss)

class AEResults:
    def __init__(self, cost):
        self.cost = cost

    # given several val_infos, combines all data into one val_info
    @staticmethod
    def joinValInfos(all_val_info):
        avg_cost = np.mean([vi.cost for vi in all_val_info])

        return AEResults(cost=avg_cost)

    def compare(self, other):
        if self.cost < other.cost:
            print("new best score", self.cost)
            return 1
        elif self.cost == other.cost:
            return 0
        else:
            return -1

    def printStatistics(self):
        print("cost %.3f" % self.cost)

    def truth_pred(self):
        return np.zeros((1,1))

class ClassifierResults:
    def __init__(self, y, pred_y, cost):
        self.y = y
        if len(pred_y.shape) == 1:
            self.pred_y = pred_y.reshape(-1, 1)
        else:
            self.pred_y = pred_y
        self.cost = cost

    # given several val_infos, combines all data into one val_info
    @staticmethod
    def joinValInfos(all_val_info):
        ally = np.concatenate([ vi.y for vi in all_val_info ])
        all_pred_y = np.concatenate([ vi.pred_y for vi in all_val_info ])

        avg_cost = np.mean([vi.cost for vi in all_val_info])

        return ClassifierResults(y=ally, pred_y=all_pred_y, cost=avg_cost)

    def compare(self, other):
        if self.cost < other.cost:
            print("new best score", self.cost)
            return 1
        elif self.cost == other.cost:
            return 0
        else:
            return -1

    def printStatistics(self):
        print("cost %.3f" % self.cost)
        print("accuracy %.3f precision %.3f recall %.3f" % \
            (self.accuracy_score(), \
            self.precision_score(), \
            self.recall_score()))

    def accuracy_score(self):
        return np.mean([accuracy_score(self.y, self.pred_y[:,i]) \
            for i in range(self.pred_y.shape[1])])

    def precision_score(self):
        return np.mean([precision_score(self.y, self.pred_y[:,i], average='weighted') \
            for i in range(self.pred_y.shape[1])])

    def recall_score(self):
        return np.mean([recall_score(self.y, self.pred_y[:,i], average='weighted') \
            for i in range(self.pred_y.shape[1])])

    def f1(self):
        return np.mean([f1_score(self.y, self.pred_y[:,i], average='weighted') \
            for i in range(self.pred_y.shape[1])])

    def truth_pred(self):
        return np.column_stack((self.y, self.pred_y))

    # given one additional val_info, concatenates it along axis=1.
    # used for running several dropout masks for the same dataset
    def concatPredictions(self, other_val_info):
        self.pred_y = np.concatenate([self.pred_y, other_val_info.pred_y], axis=1)

def countClasses(labels):
    unique = set()
    for l in labels:
        unique.add(l)

    return len(unique)

def printPerformance(truth, predicted, cross_entropy=0):
    binary = not(countClasses(truth)>2 or countClasses(predicted)>2)
    if binary:
        average = 'binary'
    else:
        average = 'weighted'

    print ("len truth", len(truth), "len predicted", len(predicted))

    acc = accuracy_score(truth, predicted)
    print("accuracy_score", acc)
    print("f1_score", f1_score(truth, predicted, average=average))
    print("precision_score", precision_score(truth, predicted, average=average))
    print("recall_score", recall_score(truth, predicted, average=average))
    print("r2_score", r2_score(truth, predicted))
    print("cross_entropy_score", cross_entropy)

    return acc

def printLabelStats(allY):
    labels = np.argmax(allY, axis=1)
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 1
        else:
            counts[label] += 1

    class_counts = sorted(counts.iteritems())
    total = sum([c[1] for c in class_counts])

    print("total number of samples", total)
    for c in class_counts:
        print("class", c[0], "\tcount: ", c[1], "\tpercent:", c[1]/float(total))
