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
#import matplotlib.pyplot as plt

from scipy import misc
from tf_dataset import VariableSet

from tf_validationInfo import RSquaredResults
from tf_graph import *
import tf_encode
from tf_MNIST import MNIST, MNIST_val
import tf_log

from sklearn.metrics import accuracy_score, f1_score, \
                        precision_score, recall_score, \
                        r2_score

import argparse
import timeit
import time

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class PredictionBuffers:
    def __init__(self, nameA, nameB):
        self.a = nameA
        self.b = nameB

    def print(self, original, predicted):
        #print("original shape", original.shape)
        #print("predicted shape", predicted.shape)
        joined = np.concatenate((original, predicted), 1)
        #print("joinded shape", joined)

        np.savetxt(self.a, joined, "%.1f", delimiter="\t")

        self.a, self.b = self.b, self.a

def list_of_strs(a):
    if type(a) == type(''):
        return [a]
    else:
        return a

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-X', '--trainx', nargs='+', default=[''], action='store', help='path to training samples')
    parser.add_argument('-Z', '--trainz', nargs='+', default=[''], action='store', help='secondary training feature samples')
    parser.add_argument('-D', '--traind', nargs='+', default=[''], action='store', help='tertiary training feature samples')
    parser.add_argument('-y', '--trainy', nargs='+', default=[''], action='store', help='path to training labels')
    parser.add_argument('-v', '--valid', nargs='+', default=[''], action='store', help='path to validation data')
    parser.add_argument('-vz', '--validz', nargs='+', default=[''], action='store', help='secondary validation feature samples')
    parser.add_argument('-vd', '--validd', nargs='+', default=[''], action='store', help='tertiary validation feature samples')
    parser.add_argument('-vy', '--validy', nargs='+', default=[''], action='store', help='path to validation labels')

    parser.add_argument('-o', '--outputFolder', default='', action='store', help='path to output folder')
    parser.add_argument('-s', '--summaryFolder', default='', action='store', help='path to output folder for summaries')
    parser.add_argument('-vs', '--valid_size', default=900, type=int, action='store', help='how much of the validation to use.')
    parser.add_argument('-bs', '--batch_size', default=50, type=int, action='store', help='mini batch size for training')
    parser.add_argument('-ec', '--epoch_count', default=160, type=int, action='store', help='number of training epochs')
    parser.add_argument('-ds', '--display_step', default=5, type=int, action='store', help='number of times you display per epoch')
    parser.add_argument('-kp', '--keep_prob', default=1, type=float, action='store', help='keep_prob used for dropout')
    parser.add_argument('-lr', '--learning_rate', default=.0001, type=float, action='store', help='learning rate used for optimization')
    parser.add_argument('-ae', '--drug_type', default='', action='store', help='which AutoEncoder model to use')
    parser.add_argument('-rae', '--rna_ae_type', default='', action='store', help='which AutoEncoder model for RNA to use')
    parser.add_argument('-ot', '--graph_type', default='', action='store', help='changes the cost function type')
    parser.add_argument('--scaleModel', default='', action='store', type=str, help='if you have a saved normalizing model, insert it here')
    parser.add_argument('--weight_decay', default=0, action='store', type=float, help='scale the weight decay regularization')
    parser.add_argument('--out_check_name', default='auto_convnet.ckpt', action='store', help='name of the checkpoint file')
    parser.add_argument('--pretrained_weights', default=None, action='store', help='for loading checkpoints')
    parser.add_argument('--epochShuffle', default=False, type=bool, action='store', help='this always happens now.')
    parser.add_argument('--cost', default='', type=str, action='store', help='cost function type. rms or cross_entropy')
    parser.add_argument('--bilinear', action='store_true', help='using compact bilinear layer to combine two y branches')
    parser.add_argument('--encode', action='store_true', default=False, help='instead of predicting, just encode')
    parser.add_argument('--center_loss_alpha', action='store', default=0., type=float, help='alpha scales how quickly centers chagne?')
    parser.add_argument('--center_loss_lambdas', nargs=3, type=float, help='scales center_loss, reconstruction, cross entropy, penalty, normalized internally')
    parser.add_argument('--weight_dump', nargs='+', default=[], help='list of wehights that you want to save')
    parser.add_argument('--dist_weight', type=float, default=0., help='weight for cost function that measures "distance from the line"')
    parser.add_argument('--keep_mask_loops', type=int, default=1, help='number of validation loops with different dropout masks')
    parser.add_argument('--prediction_suffix', type=str, default="_pred", help="the suffix appended to the end of prediction output names")
    parser.add_argument('-CUDA_VISIBLE_DEVICES', type=str, default='', help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    print(args)

    return args

def checkFilesExist(args):
    files = args.trainx+args.trainz+args.traind+args.trainy+args.valid+args.validy+args.validz+args.validd

    print("files", files)
    for f in files:
        if not f == '' and not os.path.exists(f):
            print("couldn't find", f)
            sys.exit(1)

# sets up the training and validation dataset pairs
def buildDatasets(args):
    checkFilesExist(args)
    return getOp(args).buildTrainDataset(args)

# sets up datasets for validation/test:
def buildValDatasets(args):
    checkFilesExist(args)
    return getOp(args).buildValidationDataset(args)

# sets up datasets for per chunk output, encoding for example:
def buildSplitDatasets(args):
    checkFilesExist(args)
    return getOp(args).buildSplitDataset(args)

def makeSummaryDirectories(args):
    trainPath = os.path.join(args.summaryFolder, args.out_check_name+'_train')
    valPath = os.path.join(args.summaryFolder, args.out_check_name+'_val')

    if not os.path.exists(trainPath):
        os.makedirs(trainPath)

    if not os.path.exists(valPath):
        os.makedirs(valPath)

    return trainPath, valPath

def writeCheckpointVars(path):
    with open(path, 'w') as varFile:
        print("trainable variables")
        variableNames = [t.name for t in trainableVariables]
        print(variableNames)
        for name in variableNames:
            varFile.write(name+"\n")

    sys.exit(1)

def set_cuda_visible_devices(args):
    if len(args.CUDA_VISIBLE_DEVICES) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

def trainModels(args):
    trainData, valData = buildDatasets(args)

    bestPerf = None
    set_cuda_visible_devices(args)
    with tf.Graph().as_default():
        # check for pretrained weights
        pretrained = None
        if args.pretrained_weights:
            pretrainedPath = os.path.join(args.outputFolder, args.pretrained_weights)
            print("pretrained path", pretrainedPath)
            pretrained = VariableSet(pretrainedPath)

        # check for checkpoint to load previous run
        g_Output_CheckpointPath = os.path.join(args.outputFolder, args.out_check_name)
        ratchetCheckpoint = g_Output_CheckpointPath + "_ratchet"

        # infer model type based on presence of labeled data
        model = getOp(args)
        opGraph = model(args, trainData, 
            checkpoint=g_Output_CheckpointPath, pretrained=pretrained)

        # load training index from file if it exists
        epoch_idx_filename = os.path.join(args.outputFolder, "epoch_index.sav")
        if os.path.exists(epoch_idx_filename):
            with open(epoch_idx_filename) as load:
                start_epoch = int(load.readline().strip())
        else:
            start_epoch = 0

        # save variables that are trainable in case we're pretraining
        trainableVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        VariableSet.writeFile(trainableVariables, g_Output_CheckpointPath)

        # set up summary paths
        trainPath, valPath = makeSummaryDirectories(args)

        # setup log
        log = model.getPreferedLogger()(os.path.join(args.outputFolder, 'training.log'))
        timing_log = tf_log.String_Log(os.path.join(args.outputFolder, 'timing.log'))

        # Training cycle
        for epoch in range(start_epoch, args.epoch_count):
            # Loop over all batches
            start = timeit.default_timer()
            print("epoch", epoch)
            results = []
            loop = False
            net_times = []
            read_times = []
            while not loop:
                read_start = timeit.default_timer()
                data = trainData.get_next_batch()
                read_times.append(timeit.default_timer()-read_start)

                loop = data[-1]
                time_elapsed, result = opGraph.partialFit(data, epoch)
                net_times.append(time_elapsed)

                results.append(result)

            timing_log.log("runtime time for epoch %3.f" % (timeit.default_timer()-start))

            with open(epoch_idx_filename, 'w') as save:
                save.write(str(epoch))

            epoch_result = results[0].joinValInfos(results)

            # save network
            opGraph.saver.save(opGraph.sess, g_Output_CheckpointPath)

            timing_log.log("num loops %d" % (len(net_times)))
            timing_log.log("avg network time %.3f, avg read time %.3f" % \
                (np.mean(net_times), np.mean(read_times)))
            timing_log.log("max network time %.3f, max read time %.3f" % \
                (np.max(net_times), np.max(read_times)))
            timing_log.log("total network time %.3f, total read time %.3f" % \
                (np.sum(net_times), np.sum(read_times)))

            # occasionally validate
            if epoch % args.display_step == 0:
                vi = validateLoop(opGraph, valData)

                # save the model if the performance increses. this is the 'ratchet' mechanism
                # 1: means this validation went better than the bestPerf
                if bestPerf is None or vi.compare(bestPerf) == 1:
                    timestr = time.strftime("%Y-%m-%d %H:%M")
                    print(timestr)
                    bestPerf = vi
                    opGraph.saver.save(opGraph.sess, ratchetCheckpoint)

                valData.reset()

                log.log_result(epoch, epoch_result, vi)
            else:
                # don't report validation info if there isn't any
                log.log_result(epoch, epoch_result)

            valData.reset()
            trainData.reset()
            trainData.randomize()


        print("Optimization Finished!")

def validateModel(args):
    g_Output_CheckpointPath = os.path.join(args.outputFolder, args.out_check_name)

    valData = buildSplitDatasets(args)

    print("there are", valData[0].numFeatures, "features")

    set_cuda_visible_devices(args)
    with tf.Graph().as_default():
        # restore saved weights if available
        if os.path.exists(g_Output_CheckpointPath+'.index'):
            print("Found checkpoint", g_Output_CheckpointPath)
            checkpoint = g_Output_CheckpointPath
        else:
            print("checkpoint not found", g_Output_CheckpointPath)
            return

        # build network
        model = getOp(args)
        opGraph = model(args, valData[0], checkpoint=g_Output_CheckpointPath)

        chunk_preds = []
        for chunk in valData:
            combined_preds = None
            print("keep_mask_loops", args.keep_mask_loops)
            for mask in range(args.keep_mask_loops):
                print("mask loop %d" % mask)
                mask_pred = validateLoop(opGraph, chunk)
                if combined_preds is None:
                    combined_preds = mask_pred
                else:
                    combined_preds.concatPredictions(mask_pred)
                chunk.reset()

            current_label_name = chunk.current_label_name()
            # make the filename for the dump
            dump_filename = os.path.basename(current_label_name)
            dump_filename = os.path.splitext(dump_filename)[0]+args.prediction_suffix+".csv"
            dump_filename = os.path.join(args.outputFolder, dump_filename)

            print("dumping to", dump_filename)
            np.savetxt(dump_filename, combined_preds.truth_pred(), fmt='%.4f')

            chunk_preds.append(combined_preds)

        joined_results = chunk_preds[0].joinValInfos(chunk_preds)
        joined_results.printStatistics()

        log = model.getPreferedLogger()(os.path.join(args.outputFolder, 'testing.log'))
        log.log_result(0, joined_results)

# for use in training loop. doesn't save any predictions anywhere
def validateLoop(opGraph, valData):
    stop = False
    all_val_info = []
    current_label_name = valData.current_label_name()

    while not stop:
        data = valData.get_onetime_batch()
        stop = data[-1]

        val_info = opGraph.validate(data)

        all_val_info.append(val_info)

    return all_val_info[0].joinValInfos(all_val_info)

def dumpWeights(args):
    g_Output_CheckpointPath = os.path.join(args.outputFolder, args.out_check_name)

    valData = buildValDatasets(args)
    print("there are", valData.numFeatures, "features")

    set_cuda_visible_devices(args)
    with tf.Graph().as_default():
        # restore saved weights if available
        if os.path.exists(g_Output_CheckpointPath+'.index'):
            print("Found checkpoint", g_Output_CheckpointPath)
            checkpoint = g_Output_CheckpointPath
        else:
            print("checkpoint not found", g_Output_CheckpointPath)
            return

        # infer model type based on presence of labeled data
        model = getOp(args)

        opGraph = model(args, valData, checkpoint=g_Output_CheckpointPath)

        # print out all the trainable vars
        for w in args.weight_dump:
            print("looking for", w)
            weights = None
            biases = None
            for v in tf.trainable_variables():
                if w in v.name and 'weights' in v.name:
                    print("found", v.name)
                    weights = v
                if w in v.name and 'biases' in v.name:
                    print("found", v.name)
                    biases = v
            weights, biases = opGraph.sess.run([weights, biases])

            weight_out = os.path.join(args.outputFolder, w+"_weight.npy")
            bias_out = os.path.join(args.outputFolder, w+"_bias.npy")

            np.save(weight_out, weights)
            np.save(bias_out, biases)

if __name__ == "__main__":
    args = parseArgs()
    print("weight dump", args.weight_dump)
    print("len weight dump", len(args.weight_dump))

    if len(args.weight_dump) != 0:
        dumpWeights(args)
    elif args.encode:
        tf_encode.encodeFeatures(args)
    elif args.trainx[0] == '':
        validateModel(args)
    else:
        trainModels(args)

