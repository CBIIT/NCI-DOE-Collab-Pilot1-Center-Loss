from __future__ import division, print_function, absolute_import

import sys
import os

import tensorflow as tf
import numpy as np
from tf_layers import linear_layer
import tf_dataset as tfd
from tf_model import getNetwork
from center_loss import *
import tf_log

import sklearn.metrics

from tf_validationInfo import RSquaredResults, CenterLossAEResults, ClassificationAutoencoderResults, \
    ClassifierResults, AEResults

import timeit

def getOp(args):
    t = args.graph_type
    if t == 'DosageDrugRegression':
        print("loading DosageDrugRegression")
        return DosageDrugRegression
    elif t == 'AUCRegression':
        print("loading AUCRegression")
        return AUCRegression
    elif t == 'ClassificationAutoencoder':
        return ClassificationAutoencoder
    elif t == 'TumorClassifier':
        return TumorClassifier
    elif t == 'TumorAE':
        return TumorAE
    elif t == 'CenterLossAutoencoder':
        return CenterLossAutoencoder
    elif t == 'CenterLossAutoencoder40':
        return CenterLossAutoencoder40
    elif t == 'ClassificationAutoencoder40':
        return ClassificationAutoencoder40
    elif t == 'TumorClassifier40':
        return TumorClassifier40
    elif t == 'TumorAE40':
        return TumorAE40
    elif t == 'CenterLossAutoencoder20':
        return CenterLossAutoencoder20
    elif t == 'ClassificationAutoencoder20':
        return ClassificationAutoencoder20
    elif t == 'TumorClassifier20':
        return TumorClassifier20
    elif t == 'TumorAE20':
        return TumorAE20
    else:
        print("unrecognized type", t)

class Network(object):
    def __init__(self, args, dataset, checkpoint=None, pretrained=None):
        self.args = args
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.pretrained = pretrained

        self.makePlaceholders()
        self.makeNetwork()
        self.makeCostFunction()
        self.makeOptimizer()

        # Initializing the variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.loadWeights()

    def loadWeights(self):
        print("checkpoint is", self.checkpoint)
        #tensorflow saver.
        if os.path.exists(self.checkpoint+".index"):
            # check if resuming previous run
            print("restoring from previous run")
            self.saver = tf.train.Saver(tf.global_variables())

            self.saver.restore(self.sess, self.checkpoint)
        else:
            # default create saver with all variables.
            self.saver = tf.train.Saver(tf.global_variables())

class EncodeDecodeOriginal:
    def encoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector

        self.dropout_test = tf.reduce_mean(tf.layers.dropout(self.X, rate=1-self.keep_prob))

        fc1 = tf.layers.dense(self.X, 
            5000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc1")

        fc2 = tf.layers.dense(tf.layers.dropout(fc1, rate=1-self.keep_prob), 
            3000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc2")

        fc3 = tf.layers.dense(tf.layers.dropout(fc2, rate=1-self.keep_prob), 
            2000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc3")

        fc4 = tf.layers.dense(tf.layers.dropout(fc3, rate=1-self.keep_prob), 
            943, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc4")

        return fc4

    def decoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector
        de_fc1 = tf.layers.dense(tf.layers.dropout(self.encoded, rate=1-self.keep_prob), 
            2000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc1")

        de_fc2 = tf.layers.dense(tf.layers.dropout(de_fc1, rate=1-self.keep_prob), 
            3000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc2")

        de_fc3 = tf.layers.dense(tf.layers.dropout(de_fc2, rate=1-self.keep_prob), 
            5000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc3")

        de_fc4 = tf.layers.dense(de_fc3, self.dataset.numFeatures,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            activation=tf.nn.relu,
            name="de_fc4")

        return de_fc4

class EncodeDecode40:
    def encoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector
        fc1 = tf.layers.dense(self.X, 
            5000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc1")

        fc2 = tf.layers.dense(tf.layers.dropout(fc1, rate=1-self.keep_prob), 
            1000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc2")

        fc3 = tf.layers.dense(tf.layers.dropout(fc2, rate=1-self.keep_prob), 
            200, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc3")

        fc4 = tf.layers.dense(tf.layers.dropout(fc3, rate=1-self.keep_prob), 
            40, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc4")

        return fc4

    def decoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector
        de_fc1 = tf.layers.dense(tf.layers.dropout(self.encoded, rate=1-self.keep_prob), 
            200, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc1")

        de_fc2 = tf.layers.dense(tf.layers.dropout(de_fc1, rate=1-self.keep_prob), 
            1000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc2")

        de_fc3 = tf.layers.dense(tf.layers.dropout(de_fc2, rate=1-self.keep_prob), 
            5000, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc3")

        de_fc4 = tf.layers.dense(de_fc3, self.dataset.numFeatures,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            activation=tf.nn.relu,
            name="de_fc4")

        return de_fc4

class EncodeDecode20:
    def encoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector
        fc1 = tf.layers.dense(self.X, 
            2500, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc1")

        fc2 = tf.layers.dense(tf.layers.dropout(fc1, rate=1-self.keep_prob), 
            500, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc2")

        fc3 = tf.layers.dense(tf.layers.dropout(fc2, rate=1-self.keep_prob), 
            100, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc3")

        fc4 = tf.layers.dense(tf.layers.dropout(fc3, rate=1-self.keep_prob), 
            20, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="fc4")

        return fc4

    def decoder(self):
        # for MNIST. expects self.X to have 28^2 flat input vector
        de_fc1 = tf.layers.dense(tf.layers.dropout(self.encoded, rate=1-self.keep_prob), 
            100, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc1")

        de_fc2 = tf.layers.dense(tf.layers.dropout(de_fc1, rate=1-self.keep_prob), 
            500, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc2")

        de_fc3 = tf.layers.dense(tf.layers.dropout(de_fc2, rate=1-self.keep_prob), 
            2500, activation=tf.nn.selu,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            name="de_fc3")

        de_fc4 = tf.layers.dense(de_fc3, self.dataset.numFeatures,
            kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'),
            activation=tf.nn.relu,
            name="de_fc4")

        return de_fc4

class ClassificationAutoencoder__(Network):
    @staticmethod
    def buildTrainDataset(args):
        # check to see which chunk we left off on
        trainData = tfd.ChunkGroup(pathFeatures=[args.trainx], 
            pathLabels=args.trainy, batch_size=args.batch_size,
            preprocessing_fn=os.path.join(args.outputFolder, 'onehot_labelencoder.pkl'))

        valData = ClassificationAutoencoder.buildValidationDataset(args)
        return trainData, valData

    @staticmethod
    def buildValidationDataset(args):
        valData = tfd.ChunkGroup(pathFeatures=[args.valid], 
            pathLabels=args.validy, batch_size=args.valid_size, shuffle=False,
            preprocessing_fn=os.path.join(args.outputFolder, 'onehot_labelencoder.pkl'))

        return valData

    @staticmethod
    def buildSplitDataset(args):
        pairs = zip(args.valid, args.validy)

        datasets = []
        for x, y in pairs:
            datasets.append(tfd.ChunkGroup(pathFeatures=[[x]], 
                pathLabels=[y], batch_size=args.valid_size, shuffle=False,
                preprocessing_fn=os.path.join(args.outputFolder, 'onehot_labelencoder.pkl')))

        return datasets

    @staticmethod
    def getPreferedLogger():
        return tf_log.ClassificationAutoencoderLog

    def makePlaceholders(self):
        # Placeholders for inputs
        self.X = tf.placeholder(tf.float32, [None, self.dataset.numFeatures], 
            name='ph_input_tensor')
        self.y_true = tf.placeholder(tf.int32, [None, self.dataset.numClasses], 
            name='ph_y_true')

        # drop out
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')

        # can be used for exponential decay learning rate
        self.epoch = tf.placeholder(tf.int32, name='ph_epoch')
        self.learning_rate = tf.placeholder(tf.float32, name='ph_learning_rate')

    def makeNetwork(self):
        self.encoded = self.encoder()
        self.y_pred = self.decoder()

        self.logits = tf.layers.dense(self.encoded, self.dataset.numClasses)
        indicies = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        self.class_prediction = tf.one_hot(indicies, self.dataset.numClasses)

    def makeCostFunction(self):
        # Targets (Labels) are the input data.
        # Define loss and optimizer, minimize the squared error
        self.reconstructionError = tf.sqrt(tf.reduce_mean(tf.pow(self.X - self.y_pred, 2)))
        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.y_true))

        self.cost = (1-self.args.center_loss_alpha)*self.reconstructionError + \
            self.args.center_loss_alpha*self.cross_entropy_loss

    def makeOptimizer(self):
        #decayedLearn = tf.train.exponential_decay(self.learning_rate, \
                            #self.epoch, 1, .9, staircase=True)
        #tf.summary.scalar("decayedLearn", decayedLearn)

        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = opt.minimize(self.cost)

    def partialFit(self, data, epoch):
        X, y, loop = data
        feed_dict={self.X : X, self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate, \
                    self.y_true : y}

        start = timeit.default_timer()
        cost, y_pred, class_pred, opt, = self.sess.run([self.cost, \
                        self.y_pred, self.class_prediction, self.optimizer \
                        ], feed_dict)

        return timeit.default_timer()-start, \
            ClassificationAutoencoderResults(X=X, y=self.dataset.inverse_transform(y), 
            pred_X=y_pred, pred_y=self.dataset.inverse_transform(class_pred), 
            cost=cost)

    def validate(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0, \
                    self.y_true : y}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        cost, y_recon, class_pred = \
                self.sess.run([self.cost, \
                self.y_pred, \
                self.class_prediction], \
                feed_dict)

        return ClassificationAutoencoderResults(X=X, y=self.dataset.inverse_transform(y), 
            pred_X=y_recon, pred_y=self.dataset.inverse_transform(class_pred), cost=cost)

    def encode(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0, \
                    self.y_true : y}

        # ask for the activiations from the tensor saved as self.encoded
        [encoded] = self.sess.run([self.encoded] , feed_dict)

        return encoded

class ClassificationAutoencoder(ClassificationAutoencoder__, EncodeDecodeOriginal):
    pass

class ClassificationAutoencoder40(ClassificationAutoencoder__, EncodeDecode40):
    pass

class ClassificationAutoencoder20(ClassificationAutoencoder__, EncodeDecode20):
    pass

class CenterLossAutoencoder__(ClassificationAutoencoder__):
    @staticmethod
    def getPreferedLogger():
        return tf_log.CenterLossAELog

    def makeCostFunction(self):
        # center loss stuff
        # calculate binary labels for center_loss, must be in one_hot encoding
        self.center_loss, centers, self.centers_update_op = \
            calc_center_loss(self.encoded, self.y_true, self.args.center_loss_alpha)

        self.reconstruction_error = tf.sqrt(tf.reduce_mean(tf.pow(self.X - self.y_pred, 2)))
        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.y_true))

        lambdas = np.array(self.args.center_loss_lambdas)
        normed_lambdas = lambdas / np.sum(lambdas)
        l = normed_lambdas[0]
        l2 = normed_lambdas[1]
        l3 = normed_lambdas[2]

        self.cost = l * self.center_loss + \
            l2 * self.reconstruction_error + \
            l3 * self.cross_entropy_loss

    def makeOptimizer(self):
        #decayedLearn = tf.train.exponential_decay(self.learning_rate, \
                            #self.epoch, 1, .9, staircase=True)
        #tf.summary.scalar("decayedLearn", decayedLearn)

        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = opt.minimize(self.cost)

    def partialFit(self, data, epoch):
        X, y, loop = data
        feed_dict={self.X : X, self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate, \
                    self.y_true : y}

        start = timeit.default_timer()
        cost, pred_X, class_pred, \
        center_loss, \
        cuop, op = self.sess.run([self.cost, \
            self.y_pred, self.class_prediction, \
            self.center_loss,
            self.centers_update_op, self.optimizer \
            ], feed_dict)

        return timeit.default_timer()-start, \
            CenterLossAEResults(X=X, y=self.dataset.inverse_transform(y), 
            pred_X=pred_X, pred_y=self.dataset.inverse_transform(class_pred), 
            cost=cost, center_loss=center_loss)

    def validate(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0, \
                    self.y_true : y}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        cost, pred_X, class_pred, \
        center_loss = self.sess.run([self.cost, \
            self.y_pred, self.class_prediction, \
            self.center_loss,
            ], feed_dict)

        return CenterLossAEResults(X=X, y=self.dataset.inverse_transform(y), 
            pred_X=pred_X, pred_y=self.dataset.inverse_transform(class_pred), 
            cost=cost, center_loss=center_loss)

class CenterLossAutoencoder(CenterLossAutoencoder__, EncodeDecodeOriginal):
    pass

class CenterLossAutoencoder40(CenterLossAutoencoder__, EncodeDecode40):
    pass

class CenterLossAutoencoder20(CenterLossAutoencoder__, EncodeDecode20):
    pass

class TumorAE__(ClassificationAutoencoder__):
    @staticmethod
    def getPreferedLogger():
        return tf_log.AELog

    def makePlaceholders(self):
        # Placeholders for inputs
        self.X = tf.placeholder(tf.float32, [None, self.dataset.numFeatures], name='ph_input_tensor')

        # drop out
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')

        # can be used for exponential decay learning rate
        self.epoch = tf.placeholder(tf.int32, name='ph_epoch')
        self.learning_rate = tf.placeholder(tf.float32, name='ph_learning_rate')

    def makeNetwork(self):
        self.encoded = self.encoder()
        self.y_pred = self.decoder()

    def makeCostFunction(self):
        self.reconstructionError = tf.sqrt(tf.reduce_mean(tf.pow(self.X - self.y_pred, 2)))
        self.cost = self.reconstructionError

    def partialFit(self, data, epoch):
        X, y, loop = data
        feed_dict={self.X : X, self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate}

        start = timeit.default_timer()
        cost, opt, = self.sess.run([self.cost, \
                        self.optimizer \
                        ], feed_dict)

        return timeit.default_timer()-start, \
            AEResults(cost=cost)

    def validate(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        cost = \
                self.sess.run([self.cost], \
                feed_dict)

        return AEResults(cost=cost)

    def encode(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0}

        # ask for the activiations from the tensor saved as self.encoded
        [encoded] = self.sess.run([self.encoded] , feed_dict)

        return encoded

class TumorAE(TumorAE__, EncodeDecodeOriginal):
    pass

class TumorAE40(TumorAE__, EncodeDecode40):
    pass

class TumorAE20(TumorAE__, EncodeDecode20):
    pass

class TumorClassifier__(ClassificationAutoencoder):
    @staticmethod
    def getPreferedLogger():
        return tf_log.ClassifierLog

    def makePlaceholders(self):
        # Placeholders for inputs
        self.X = tf.placeholder(tf.float32, [None, self.dataset.numFeatures], name='ph_input_tensor')
        # input and output labels are in one_hot encoding
        self.y_true = tf.placeholder(tf.int32, [None, self.dataset.numClasses], name='ph_y_true')

        # drop out
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')

        # can be used for exponential decay learning rate
        self.epoch = tf.placeholder(tf.int32, name='ph_epoch')
        self.learning_rate = tf.placeholder(tf.float32, name='ph_learning_rate')

    def makeNetwork(self):
        self.encoded = self.encoder()

        self.logits = tf.layers.dense(self.encoded, self.dataset.numClasses)
        indicies = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        self.class_prediction = tf.one_hot(indicies, self.dataset.numClasses)

    def makeCostFunction(self):
        # Targets (Labels) are the input data.
        # Define loss and optimizer, minimize the squared error
        self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy_loss)

    def partialFit(self, data, epoch):
        X, y, loop = data
        feed_dict={self.X : X, self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate, \
                    self.y_true : y}

        start = timeit.default_timer()
        cost, class_pred, opt, = self.sess.run([self.cost, \
                        self.class_prediction, self.optimizer \
                        ], feed_dict)

        return timeit.default_timer()-start, \
            ClassifierResults(y=self.dataset.inverse_transform(y), 
            pred_y=self.dataset.inverse_transform(class_pred), cost=cost)

    def validate(self, data):
        X, y, loop = data

        feed_dict={self.X : X, \
                    self.learning_rate : 0, self.epoch : 0, \
                    self.y_true : y}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        cost, class_pred, dropout_mean = \
                self.sess.run([self.cost, \
                self.class_prediction,
                self.dropout_test], \
                feed_dict)

        print("dropout mean", dropout_mean)

        return ClassifierResults(y=self.dataset.inverse_transform(y), 
            pred_y=self.dataset.inverse_transform(class_pred), cost=cost)

class TumorClassifier(TumorClassifier__, EncodeDecodeOriginal):
    pass

class TumorClassifier40(TumorClassifier__, EncodeDecode40):
    pass

class TumorClassifier20(TumorClassifier__, EncodeDecode20):
    pass

class DosageDrugRegression(Network):
    @staticmethod
    def buildTrainDataset(args):
        # check to see which chunk we left off on
        trainData = tfd.ChunkGroup(pathFeatures=[args.trainx, args.trainz, args.traind], 
            pathLabels=args.trainy, batch_size=args.batch_size, 
            preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))

        valData = DosageDrugRegression.buildValidationDataset(args)
        return trainData, valData

    @staticmethod
    def buildValidationDataset(args):
        if type(args.valid) == type([]):
            valData = tfd.ChunkGroup(pathFeatures=[args.valid, args.validz, args.validd], 
                pathLabels=args.validy, batch_size=args.valid_size, shuffle=False, 
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))
        elif type(args.valid) == type(''):
            valData = tfd.ChunkGroup(
                pathFeatures=[[args.valid], [args.validz], [args.validd]], 
                pathLabels=[args.validy], batch_size=args.valid_size, 
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))
        else:
            print("unrecognized input type", type(args.valid))
            valData = None

        return valData

    @staticmethod
    def buildSplitDataset(args):
        raise Exception('have not tested this split dataset method')
        pairs = zip(args.valid, args.validz, args.validd, args.validy)

        datasets = []
        for x, z, d, y in pairs:
            datasets.append(tfd.ChunkGroup(pathFeatures=[[x], [z], [d]], 
                pathLabels=[y], batch_size=args.valid_size, shuffle=False,
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl')))

        return datasets

    @staticmethod
    def getPreferedLogger():
        return tf_log.RegressionLog

    def makeOptimizer(self):
        #decayedLearn = tf.train.exponential_decay(self.learning_rate, \
                            #self.epoch, 1, .9, staircase=True)
        #tf.summary.scalar("decayedLearn", decayedLearn)

        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = opt.minimize(self.cost)

        #for grad, var in opt.compute_gradients(self.cost):
        #    if grad is not None:
        #        print("var.opt.name", var.op.name)
        #        tf.summary.histogram(var.op.name+'/gradients', grad)
        #        #pass

    def makePlaceholders(self):
        self.drug_X = tf.placeholder(tf.float32, 
            [None, self.dataset.features[0].numFeatures], name='ph_input_drugs')
        self.rnaseq_X = tf.placeholder(tf.float32, 
            [None, self.dataset.features[1].numFeatures], name='ph_input_rnaseq')
        self.dose_X = tf.placeholder(tf.float32, 
            [None, 1], name='ph_input_dose')

        self.learning_rate = tf.placeholder(tf.float32, name='ph_learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')
        self.epoch = tf.placeholder(tf.int32, name='ph_epoch')

        self.y_true = tf.placeholder(tf.float32, [None, 1], 
            name='ph_y_true')

    def makeNetwork(self):
        self.drug_ae = getNetwork(self.args.drug_type)()
        self.rnaseq_ae = getNetwork(self.args.rna_ae_type)()

        # rnaseq autoencoder
        self.rna_encoded = self.rnaseq_ae.encoder(self.rnaseq_X, 
            self.keep_prob, self.args.weight_decay)

        self.drug_encoded = self.drug_ae.encoder(self.drug_X, 
            self.keep_prob, self.args.weight_decay)

        print("using full connected layer to join branches")
        print("second_encode", self.drug_encoded.get_shape().as_list())
        print("self.rnaseq_ae", self.rna_encoded.get_shape().as_list())
        joined = tf.concat([self.drug_encoded, self.rna_encoded, self.dose_X], 
            1, name='branch_concat')

        classL1 = linear_layer(joined, "classL1", 1000, \
                                    stddev=.01, wd=self.args.weight_decay)

        classL2 = linear_layer(tf.nn.dropout(classL1, self.keep_prob), "classL2", 1000, \
                                    stddev=.01, wd=self.args.weight_decay)

        classL3 = linear_layer(tf.nn.dropout(classL2, self.keep_prob), "classL3", 1000, \
                                        stddev=.01, wd=self.args.weight_decay)

        self.encoded = classL3

    def makeCostFunction(self):
        #tf.summary.scalar('epoch', self.epoch)
        # compute performance of classifier
        self.y_pred = linear_layer(self.encoded, "regressorTensor", 
            1, stddev=.01, wd=self.args.weight_decay, nonlinearity=tf.identity)

        with tf.variable_scope("cost_calculations") as scope:
            # rms cost
            self.rms_error = tf.sqrt(tf.reduce_mean(
                tf.pow(self.y_pred - self.y_true, 2)))

            tf.add_to_collection('losses', self.rms_error)

            losses_collection = tf.get_collection('losses')
            for l in losses_collection:
                print("loss:", l.name)

            self.cost = tf.add_n(losses_collection)
            #tf.summary.scalar('cost', self.cost)
            #tf.summary.scalar('weight decay', self.cost - self.rms_error)

    def partialFit(self, data, epoch):
        X, Z, D, y, loop = data
        D = D.reshape((D.shape[0],1))
        y = y.reshape((y.shape[0],1))

        start = timeit.default_timer()
        feed_dict={self.drug_X : X, self.rnaseq_X: Z, self.dose_X: D,
                    self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate, \
                    self.y_true : y}

        # necessary to run
        [cost, regression, opt] = self.sess.run([ \
                self.cost, self.y_pred, self.optimizer, 
            ], feed_dict)

        return timeit.default_timer()-start, RSquaredResults(Y=y, cost=cost, P=regression)

    def validate(self, data):
        X, Z, D, y, loop = data
        D = D.reshape((D.shape[0],1))
        y = y.reshape((y.shape[0],1))

        feed_dict={self.drug_X : X, self.rnaseq_X : Z, self.dose_X : D,
                    self.learning_rate : 0, self.epoch : 0, \
                    self.y_true : y}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        # necessary to run
        [cost, regression] = self.sess.run([ \
                self.cost, self.y_pred,
            ], feed_dict)

        return RSquaredResults(Y=y, cost=cost, P=regression)

    def classify(self, data):
        val_info = self.validate(data)
        return val_info

    def encode(self, data):
        D = data[2]
        D = D.reshape((D.shape[0],1))

        feed_dict = {
            self.drug_X : data[0], self.rnaseq_X : data[1], self.dose_X : D,
            self.keep_prob : 1,
            self.learning_rate : 0, self.epoch : 0,
            # make some fake labels, we don't care
            self.y_true : np.zeros((len(data[0]), self.dataset.numClasses))
        }

        # ask for the activiations from the tensor saved as self.encoded
        [encoded] = self.sess.run([self.encoded] , feed_dict)

        return encoded

class AUCRegression(Network):
    @staticmethod
    def buildTrainDataset(args):
        # check to see which chunk we left off on
        trainData = tfd.KeyedChunkGroup(pathFeatures=[args.trainx, args.trainz], 
            pathLabels=args.trainy, batch_size=args.batch_size, 
            preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))

        valData = AUCRegression.buildValidationDataset(args)

        return trainData, valData

    @staticmethod
    def buildValidationDataset(args):
        if type(args.valid) == type([]):
            valData = tfd.KeyedChunkGroup(pathFeatures=[args.valid, args.validz], 
                pathLabels=args.validy, batch_size=args.valid_size, shuffle=False, 
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))
        elif type(args.valid) == type(''):
            valData = tfd.KeyedChunkGroup(pathFeatures=[[args.valid], [args.validz]], 
                pathLabels=[args.validy], batch_size=args.valid_size,
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl'))
        else:
            print("unrecognized input type", type(args.valid))
            valData = None

        return valData

    @staticmethod
    def buildSplitDataset(args):
        raise Exception('have not tested this split dataset method')
        pairs = zip(args.valid, args.validz, args.validy)

        datasets = []
        for x, z, y in pairs:
            datasets.append(tfd.ChunkGroup(pathFeatures=[[x], [z]], 
                pathLabels=[y], batch_size=args.valid_size, shuffle=False,
                preprocessing_fn=os.path.join(args.outputFolder, 'no_preprocessor.pkl')))

        return datasets

    @staticmethod
    def getPreferedLogger():
        return tf_log.RegressionLog

    def makePlaceholders(self):
        self.drug_X = tf.placeholder(tf.float32, 
            [None, self.dataset.features[0].numFeatures], name='ph_input_drugs')
        self.rnaseq_X = tf.placeholder(tf.float32, 
            [None, self.dataset.features[1].numFeatures], name='ph_input_rnaseq')

        self.learning_rate = tf.placeholder(tf.float32, name='ph_learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')
        self.epoch = tf.placeholder(tf.int32, name='ph_epoch')

        self.auc_true = tf.placeholder(tf.float32, [None, 1], 
            name='ph_auc_true')

    def makeOptimizer(self):
        #decayedLearn = tf.train.exponential_decay(self.learning_rate, \
                            #self.epoch, 1, .9, staircase=True)

        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = opt.minimize(self.cost)

    def makeNetwork(self):
        self.drug_branch = getNetwork(self.args.drug_type)()
        self.rnaseq_branch = getNetwork(self.args.rna_ae_type)()

        # drug branch
        self.drug_encoded = self.drug_branch.encoder(self.drug_X, 
            self.keep_prob, self.args.weight_decay)

        # rnaseq branch
        self.rna_encoded = self.rnaseq_branch.encoder(self.rnaseq_X, 
            self.keep_prob, self.args.weight_decay)

        print("using full connected layer to join branches")
        print("second_encode", self.drug_encoded.get_shape().as_list())
        print("self.rnaseq_branch", self.rna_encoded.get_shape().as_list())
        joined = tf.concat([self.drug_encoded, self.rna_encoded], 
            1, name='branch_concat')

        classL1 = linear_layer(joined, "classL1", 1000, \
                                    stddev=.01, wd=self.args.weight_decay)

        classL2 = linear_layer(tf.nn.dropout(classL1, self.keep_prob), "classL2", 1000, \
                                    stddev=.01, wd=self.args.weight_decay)

        classL3 = linear_layer(tf.nn.dropout(classL2, self.keep_prob), "classL3", 1000, \
                                        stddev=.01, wd=self.args.weight_decay)

        self.latent = classL3

    def makeCostFunction(self):
        # compute performance of classifier
        self.auc_pred = linear_layer(self.latent, "regressorTensor", 
            1, stddev=.01, wd=self.args.weight_decay, nonlinearity=tf.identity)

        with tf.variable_scope("cost_calculations") as scope:
            # rms cost
            self.rms_error = tf.sqrt(tf.reduce_mean(
                tf.pow(self.auc_pred - self.auc_true, 2)))
            tf.add_to_collection('losses', self.rms_error)

            losses_collection = tf.get_collection('losses')
            for l in losses_collection:
                print("loss:", l.name)

            self.cost = tf.add_n(losses_collection)

    def partialFit(self, data, epoch):
        X, Z, y, loop = data
        y = y.reshape((y.shape[0],1))

        feed_dict={self.drug_X : X, self.rnaseq_X: Z,
                    self.epoch : epoch, \
                    self.keep_prob : self.args.keep_prob, \
                    self.learning_rate : self.args.learning_rate, \
                    self.auc_true : y}

        start = timeit.default_timer()
        # necessary to run
        [cost, regression, opt] = self.sess.run([ \
                self.cost, self.auc_pred, self.optimizer], feed_dict)

        return timeit.default_timer()-start, RSquaredResults(Y=y, cost=cost, P=regression)

    def validate(self, data):
        X, Z, y, loop = data
        y = y.reshape((y.shape[0],1))

        feed_dict={self.drug_X : X, self.rnaseq_X : Z,
                    self.learning_rate : 0, self.epoch : 0, \
                    self.auc_true : y}

        if self.args.keep_mask_loops == 1:
            feed_dict[self.keep_prob] = 1
        else:
            feed_dict[self.keep_prob] = self.args.keep_prob

        # necessary to run
        [cost, regression] = self.sess.run([ \
                self.cost, self.auc_pred,
            ], feed_dict)

        return RSquaredResults(Y=y, cost=cost, P=regression)

    def classify(self, data):
        val_info = self.validate(data)
        return val_info

    def encode(self, data):
        feed_dict = {
            self.drug_X : data[0], self.rnaseq_X : data[1], 
            self.keep_prob : 1,
            self.learning_rate : 0, self.epoch : 0,
            # make some fake labels, we don't care
            self.auc_true : np.zeros((len(data[0]), self.dataset.numClasses))
        }

        # ask for the activiations from the tensor saved as self.latent
        [encoded] = self.sess.run([self.latent] , feed_dict)

        return encoded

