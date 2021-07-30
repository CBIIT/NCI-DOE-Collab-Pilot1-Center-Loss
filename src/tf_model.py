from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tf_layers import *
from tf_selu import selu
import math

def getNetwork(t):
    print("using AE", t)

    if t == 'wide':
        return Regression_Wide
    if t == 'encoded':
        return Regression_Encoded
    if t == 'chem':
        return AutoEncoder_Chem
    if t == 'sig':
        return AutoEncoder_Chem_Sigmoid
    if t == 'ecfp':
        return AutoEncoder_Chem_ECFP
    if t == 'ecfp_sig':
        return AutoEncoder_Chem_ECFP_sig
    if t == 'ecfp_sig_bn':
        return AutoEncoder_Chem_ECFP_sig_bn
    if t == 'flat':
        return AutoEncoder_Chem_Flat
    if t == 'ecfp_two':
        return AutoEncoder_ECFP_Two
    if t == 'ecfp_three':
        return AutoEncoder_ECFP_Three
    if t == 'ecfp_five':
        return AutoEncoder_ECFP_Five
    if t == 'ecfp_three_bn':
        return AutoEncoder_ECFP_Three_BN
    if t == 'ecfp_skinny_bn':
        return AutoEncoder_ECFP_Skinny_BN
    if t == 'ecfp_selu':
        return AutoEncoder_ECFP_SELU
    if t == 'ecfp_selu_five':
        return AutoEncoder_ECFP_SELU_Five
    if t == 'ecfp_selu_two':
        return AutoEncoder_ECFP_SELU_Two
    if t == 'rnaseq_selu':
        return RNASEQ_SELU
    if t == 'rnaseq_selu_big':
        return RNASEQ_SELU_big
    if t == 'rnaseq_relu_big':
        return RNASEQ_RELU_big
    if t == 'rnaseq_sig_big':
        return RNASEQ_Sig_big
    if t == 'rnaseq_selu_bigger':
        return RNASEQ_SELU_bigger
    if t == 'rnaseq_selu_sq':
        return RNASEQ_SELU_sq
    if t == 'rnaseq_selu_1k':
        return RNASEQ_SELU_1k
    if t == 'rnaseq_sq':
        return RNASEQ_sq
    if t == 'lbexp_selu':
        return LBEXP_SELU
    if t == 'lbexp_relu':
        return LBEXP_RELU
    if t == 'tox_relu':
        return TOX_RELU
    if t == 'tox_relu_reg':
        return TOX_RELU_REG
    if t == 'rnaseq_selu_big':
        return RNASEQ_SELU_big
    if t == 'fang_relu_dragon':
        return FANG_RELU_DRAGON
    if t == 'fang_relu_gene':
        return FANG_RELU_GENE

    print("unrecognized autoencoder type", t)

class Regression_Wide:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = batch_normalized_linear_layer(x, "fc1", 2000, stddev=1, wd=.004, test=test)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 500, stddev=1, wd=.004, test=test)
        fc3 = batch_normalized_linear_layer(fc2, "fc3", 100, stddev=1, wd=.004, test=test)
        fc3_out = linear_layer(fc3, 'fc3_out', 1, stddev=1, wd=.004, nonlinearity=None)

        return fc3_out

class Regression_Encoded:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        #fc1 = batch_normalized_linear_layer(x, "fc1", 50, stddev=1, wd=.004, test=test)
        #fc2 = batch_normalized_linear_layer(x, "fc2", 25, stddev=1, wd=.004, test=test)
        #fc3 = batch_normalized_linear_layer(fc2, "fc3", 12, stddev=1, wd=.004, test=test)
        fc3_out = linear_layer(x, 'fc3_out', 1, stddev=1, wd=.004, nonlinearity=None)

        return fc3_out

class AutoEncoder_Chem_Sigmoid:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("using the correct AE class")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 2000, stddev=.005, wd=.004, nonlinearity=tf.sigmoid)
        fc2 = linear_layer(fc1, "fc2", 500, stddev=.005, wd=.004, nonlinearity=tf.sigmoid)
        fc3_out = linear_layer(fc2, "fc3_3", 100, stddev=.005, wd=.004, nonlinearity=tf.sigmoid)

        return fc3_out

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 500, stddev=.005, wd=.004, nonlinearity=tf.sigmoid)
        de_fc2 = linear_layer(de_fc1, "de_fc2", 2000, stddev=.005, wd=.004, nonlinearity=tf.sigmoid)
        de_fc3_out = linear_layer(de_fc2, 'de_fc3_out', self.width, stddev=.005, wd=.004)

        return de_fc3_out

class AutoEncoder_Chem:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = batch_normalized_linear_layer(x, "fc1", 2000, stddev=.005, wd=.004, test=test)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 500, stddev=.005, wd=.004, test=test)
        fc3_out = linear_layer(fc2, "fc3_3", 100, stddev=.005, wd=.004)

        return fc3_out

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = batch_normalized_linear_layer(x, "de_fc1", 500, stddev=.005, wd=.004, test=test)
        de_fc2 = batch_normalized_linear_layer(de_fc1, "de_fc2", 2000, stddev=.005, wd=.004, test=test)
        de_fc3_out = linear_layer(de_fc2, 'de_fc3_out', self.width, stddev=.005, wd=.004)

        return de_fc3_out

class AutoEncoder_Chem_Flat:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("using flat network")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 100, stddev=.005, wd=.004)

        return fc1

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = linear_layer(x, 'de_fc1', self.width, stddev=.005, wd=.004)

        return de_fc1

class AutoEncoder_Chem_ECFP:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 2000, stddev=.01, wd=.004)
        fc2 = linear_layer(fc1, "fc2", 1000, stddev=.01, wd=.004)
        fc3 = linear_layer(fc2, "fc3", 500, stddev=.01, wd=.004)
        fc4 = linear_layer(fc3, "fc4", 250, stddev=.01, wd=.004)
        fc5_out = linear_layer(fc4, "fc5_out", 100, stddev=.01, wd=.004)

        return fc5_out

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 250, stddev=.01, wd=.004)
        de_fc2 = linear_layer(de_fc1, "de_fc2", 500, stddev=.01, wd=.004)
        de_fc3 = linear_layer(de_fc2, "de_fc3", 1000, stddev=.01, wd=.004)
        de_fc4 = linear_layer(de_fc3, "de_fc4", 2000, stddev=.01, wd=.004)
        de_fc5_out = linear_layer(de_fc4, 'de_fc5_out', self.width, stddev=.01, wd=.004)

        return de_fc5_out

def intense_sigmoid(X, name=None):
    return tf.nn.sigmoid(1*X)

class AutoEncoder_Chem_ECFP_sig:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_sig")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 2000, stddev=.01, wd=.004)
        fc2 = linear_layer(fc1, "fc2", 1000, stddev=.01, wd=.004)
        fc3 = linear_layer(fc2, "fc3", 500, stddev=.01, wd=.004)
        fc4 = linear_layer(fc3, "fc4", 250, stddev=.01, wd=.004)
        fc5_out = linear_layer(fc4, "fc5_out", 100, stddev=.01, wd=.004)

        return fc5_out

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 250, stddev=.01, wd=.004)
        de_fc2 = linear_layer(de_fc1, "de_fc2", 500, stddev=.01, wd=.004)
        de_fc3 = linear_layer(de_fc2, "de_fc3", 1000, stddev=.01, wd=.004)
        de_fc4 = linear_layer(de_fc3, "de_fc4", 2000, stddev=.01, wd=.004)
        de_fc5_out = linear_layer(de_fc4, 'de_fc5_out', self.width, stddev=.01, wd=.004, nonlinearity=intense_sigmoid)

        return de_fc5_out

class AutoEncoder_Chem_ECFP_sig_bn:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_sig_bn")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x):
        test = not self.is_training
        fc1 = batch_normalized_linear_layer(x, "fc1", 2000, stddev=.01, wd=.004, test=test)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 1000, stddev=.01, wd=.004, test=test)
        fc3 = batch_normalized_linear_layer(fc2, "fc3", 500, stddev=.01, wd=.004, test=test)
        fc4 = batch_normalized_linear_layer(fc3, "fc4", 250, stddev=.01, wd=.004, test=test)
        fc5_out = linear_layer(fc4, "fc5_out", 100, stddev=.01, wd=.004)

        return fc5_out

    # Building the decoder
    def decoder(self, x):
        test = not self.is_training
        de_fc1 = batch_normalized_linear_layer(x, "de_fc1", 250, stddev=.01, wd=.004, test=test)
        de_fc2 = batch_normalized_linear_layer(de_fc1, "de_fc2", 500, stddev=.01, wd=.004, test=test)
        de_fc3 = batch_normalized_linear_layer(de_fc2, "de_fc3", 1000, stddev=.01, wd=.004, test=test)
        de_fc4 = batch_normalized_linear_layer(de_fc3, "de_fc4", 2000, stddev=.01, wd=.004, test=test)
        de_fc5_out = linear_layer(de_fc4, 'de_fc5_out', self.width, stddev=.01, wd=.004, nonlinearity=intense_sigmoid)

        return de_fc5_out

class AutoEncoder_ECFP_Two:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_two")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 100, stddev=.04, wd=None)
        fc2_out = linear_layer(fc1, "fc2_out", 100, stddev=.04, wd=None)

        return fc2_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 100, stddev=.04, wd=None)
        de_fc2_out = linear_layer(de_fc1, 'de_fc2_out', self.width, stddev=.04, \
                            wd=None, nonlinearity=tf.nn.relu)

        return de_fc2_out

class AutoEncoder_ECFP_Three:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_three")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 500, stddev=.04, wd=None)
        fc2 = linear_layer(fc1, 'fc2', 250, stddev=.04, wd=None)
        fc3_out = linear_layer(fc2, "fc3_out", 100, stddev=.04, wd=None)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 250, stddev=.04, wd=None)
        de_fc2 = linear_layer(de_fc1, "de_fc2", 500, stddev=.04, wd=None)
        de_fc3_out = linear_layer(de_fc2, 'de_fc3_out', self.width, stddev=.04, \
                            wd=None, nonlinearity=tf.sigmoid)

        return de_fc3_out

class AutoEncoder_ECFP_Five:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_three")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        fc1 = linear_layer(x, "fc1", 200, stddev=.04, wd=None)
        fc2 = linear_layer(fc1, 'fc2', 150, stddev=.04, wd=None)
        fc3 = linear_layer(fc2, "fc3", 100, stddev=.04, wd=None)
        fc4 = linear_layer(fc3, "fc4", 100, stddev=.04, wd=None)
        fc5_out = linear_layer(fc4, "fc5_out", 100, stddev=.04, wd=None)

        return fc5_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        de_fc1 = linear_layer(x, "de_fc1", 100, stddev=.04, wd=None)
        de_fc2 = linear_layer(de_fc1, "de_fc2", 100, stddev=.04, wd=None)
        de_fc3 = linear_layer(de_fc2, "de_fc3", 150, stddev=.04, wd=None)
        de_fc4 = linear_layer(de_fc3, "de_fc4", 200, stddev=.04, wd=None)
        de_fc5_out = linear_layer(de_fc4, 'de_fc5_out', self.width, stddev=.04, \
                            wd=None, nonlinearity=tf.sigmoid)

        return de_fc5_out

class AutoEncoder_ECFP_SELU_Two:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_two_SELU")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 100
        fc1 = linear_layer(x, 'fc1', f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu)

        self.f2_out_width = 100
        fc2_out = linear_layer(fc1, "fc2_out", self.f2_out_width, stddev=math.sqrt(1./f1_width), \
                                    wd=weight_decay, nonlinearity=selu)

        return fc2_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 100
        de_fc1 = linear_layer(x, "de_fc1", f1_width, stddev=math.sqrt(1./self.f2_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc2_out = linear_layer(de_fc1, 'de_fc2_out', self.width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        return de_fc2_out

class AutoEncoder_ECFP_SELU_Five:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_SELU_Five")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        f1_width = 200
        fc1 = linear_layer(x, "ecfps5_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu)

        f2_width = 150
        fc2 = linear_layer(fc1, 'ecfps5_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu)

        f3_width = 100
        fc3 = linear_layer(fc2, 'ecfps5_fc3', f3_width, stddev=math.sqrt(1./f2_width), wd=weight_decay, \
                                    nonlinearity=selu)

        f4_width = 100
        fc4 = linear_layer(fc3, 'ecfps5_fc4', f4_width, stddev=math.sqrt(1./f3_width), wd=weight_decay, \
                                    nonlinearity=selu)

        self.f5_out_width = 100
        fc5_out = linear_layer(fc4, "ecfps5_fc5_out", self.f5_out_width, stddev=math.sqrt(1./f4_width), \
                                    wd=weight_decay, nonlinearity=selu)

        return fc5_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 100
        de_fc1 = linear_layer(x, "ecfps5_de_fc1", f1_width, stddev=math.sqrt(1./self.f5_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 100
        de_fc2 = linear_layer(de_fc1, "ecfps5_de_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        f3_width = 150
        de_fc3 = linear_layer(de_fc2, "ecfps5_de_fc3", f3_width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=selu)

        f4_width = 200
        de_fc4 = linear_layer(de_fc3, "ecfps5_de_fc4", f4_width, stddev=math.sqrt(1./f3_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc5_out = linear_layer(de_fc4, 'ecfps5_de_fc5_out', self.width, stddev=math.sqrt(1./f4_width), \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        return de_fc5_out

class AutoEncoder_ECFP_SELU:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_three_SELU")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 500
        fc1 = linear_layer(x, "ecfps_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        f2_width = 250
        fc2 = linear_layer(fc1, 'ecfps_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f3_out_width = 100
        fc3_out = linear_layer(fc2, "ecfps_fc3_out", self.f3_out_width, stddev=math.sqrt(1./f2_width), \
                                    wd=weight_decay, nonlinearity=selu, reuse=reuse)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 250
        de_fc1 = linear_layer(x, "ecfps_de_fc1", f1_width, stddev=math.sqrt(1./self.f3_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 500
        de_fc2 = linear_layer(de_fc1, "ecfps_de_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc3_out = linear_layer(de_fc2, 'ecfps_de_fc3_out', self.width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        print("out layer has width", self.width)
        return de_fc3_out

class RNASEQ_SELU_big:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU_big")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 5000
        fc1 = linear_layer(x, "rnasb_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        f2_width = 2000
        fc2 = linear_layer(fc1, 'rnasb_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)


        f3_width = 400
        fc3 = linear_layer(fc2, 'rnasb_fc3', f3_width, stddev=math.sqrt(1./f2_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f4_out_width = 200
        fc4_out = linear_layer(fc3, "rnasb_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 400
        de_fc1 = linear_layer(x, "rnasb_de_fc1", f1_width, stddev=math.sqrt(1./self.f4_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 2000
        de_fc2 = linear_layer(de_fc1, "rnasb_de_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        f3_width = 5000
        de_fc3 = linear_layer(de_fc2, "rnasb_de_fc3", f3_width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc4_out = linear_layer(de_fc3, 'rnasb_de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class RNASEQ_RELU_big:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_RELU_big")
        self.is_training = is_training
        self.pfx = "rnarb"

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 5000
        fc1 = linear_layer(x, self.pfx+"_fc1", f1_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        f2_width = 2000
        fc2 = linear_layer(fc1, self.pfx+'_fc2', f2_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)


        f3_width = 400
        fc3 = linear_layer(fc2, self.pfx+'fc3', f3_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        self.f4_out_width = 200
        fc4_out = linear_layer(fc3, "rnasb_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 400
        de_fc1 = linear_layer(x, self.pfx+"_de_fc1", f1_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        f2_width = 2000
        de_fc2 = linear_layer(de_fc1, self.pfx+"_de_fc2", f2_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        f3_width = 5000
        de_fc3 = linear_layer(de_fc2, self.pfx+"_de_fc3", f3_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        de_fc4_out = linear_layer(de_fc3, self.pfx+'de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class RNASEQ_Sig_big:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_Sig_big")
        self.is_training = is_training
        self.pfx = "rnasigb"

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 5000
        fc1 = linear_layer(x, self.pfx+"_fc1", f1_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.sigmoid, reuse=reuse)

        f2_width = 2000
        fc2 = linear_layer(fc1, self.pfx+'_fc2', f2_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.sigmoid, reuse=reuse)


        f3_width = 400
        fc3 = linear_layer(fc2, self.pfx+'fc3', f3_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.sigmoid, reuse=reuse)

        self.f4_out_width = 200
        fc4_out = linear_layer(fc3, "rnasb_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 400
        de_fc1 = linear_layer(x, self.pfx+"_de_fc1", f1_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.sigmoid)

        f2_width = 2000
        de_fc2 = linear_layer(de_fc1, self.pfx+"_de_fc2", f2_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.sigmoid)

        f3_width = 5000
        de_fc3 = linear_layer(de_fc2, self.pfx+"_de_fc3", f3_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.sigmoid)

        de_fc4_out = linear_layer(de_fc3, self.pfx+'de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class RNASEQ_SELU_bigger:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU_bigger")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(x, "rnasbr_fc1", 5000, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc2 = linear_layer(fc1, 'rnasbr_fc2', 2000, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc3 = linear_layer(fc2, 'rnasbr_fc3', 1000, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc4 = linear_layer(fc3, 'rnasbr_fc4', 500, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f5_out_width = 200
        fc5_out = linear_layer(fc4, "rnasbr_fc5_out", self.f5_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc5_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        de_fc1 = linear_layer(x, "rnasbr_de_fc1", 500, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc2 = linear_layer(de_fc1, "rnasbr_de_fc2", 1000, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc3 = linear_layer(de_fc2, "rnasbr_de_fc3", 2000, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc4 = linear_layer(de_fc3, "rnasbr_de_fc4", 5000, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc5_out = linear_layer(de_fc4, 'rnasbr_de_fc5_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc5_out

class RNASEQ_SELU_sq:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU_sq")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 1000
        fc1 = linear_layer(x, "rnassq_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        f2_width = 1000
        fc2 = linear_layer(fc1, 'rnassq_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)


        f3_width = 1000
        fc3 = linear_layer(fc2, 'rnassq_fc3', f3_width, stddev=math.sqrt(1./f2_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f4_out_width = 200
        fc4_out = linear_layer(fc3, "rnassq_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 1000
        de_fc1 = linear_layer(x, "rnassq_de_fc1", f1_width, stddev=math.sqrt(1./self.f4_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 1000
        de_fc2 = linear_layer(de_fc1, "rnassq_de_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        f3_width = 1000
        de_fc3 = linear_layer(de_fc2, "rnassq_de_fc3", f3_width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc4_out = linear_layer(de_fc3, 'rnassq_de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class RNASEQ_SELU_1k:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU_sq")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 1000
        fc1 = linear_layer(x, "rnas1k_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        f2_width = 1000
        fc2 = linear_layer(fc1, 'rnas1k_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)


        f3_width = 1000
        fc3 = linear_layer(fc2, 'rnas1k_fc3', f3_width, stddev=math.sqrt(1./f2_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f4_out_width = 1000
        fc4_out = linear_layer(fc3, "rnas1k_fc4_out", self.f4_out_width, stddev=math.sqrt(1./f3_width), \
                                    wd=weight_decay, nonlinearity=selu, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 1000
        de_fc1 = linear_layer(x, "rnas1k_de_fc1", f1_width, stddev=math.sqrt(1./self.f4_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 1000
        de_fc2 = linear_layer(de_fc1, "rnas1k_de_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        f3_width = 1000
        de_fc3 = linear_layer(de_fc2, "rnas1k_de_fc3", f3_width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc4_out = linear_layer(de_fc3, 'rnas1k_de_fc4_out', self.width, stddev=math.sqrt(1./f3_width), \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class RNASEQ_SELU:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 5000
        fc1 = linear_layer(x, "rnas_fc1", f1_width, stddev=math.sqrt(1./self.width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        f2_width = 1000
        fc2 = linear_layer(fc1, 'rnas_fc2', f2_width, stddev=math.sqrt(1./f1_width), wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f3_out_width = 100
        fc3_out = linear_layer(fc2, "rnas_fc3_out", self.f3_out_width, stddev=math.sqrt(1./f2_width), \
                                    wd=weight_decay, nonlinearity=selu, reuse=reuse)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 1000
        de_fc1 = linear_layer(x, "rnas_de_fc1", f1_width, stddev=math.sqrt(1./self.f3_out_width), \
                            wd=weight_decay, nonlinearity=selu)

        f2_width = 5000
        de_fc2 = linear_layer(de_fc1, "rnas_e_fc2", f2_width, stddev=math.sqrt(1./f1_width), \
                            wd=weight_decay, nonlinearity=selu)

        de_fc3_out = linear_layer(de_fc2, 'rnas_e_fc3_out', self.width, stddev=math.sqrt(1./f2_width), \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc3_out

class RNASEQ_sq:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_sq")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        f1_width = 1000
        fc1 = linear_layer(x, "rnasq_fc1", f1_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        f2_width = 1000
        fc2 = linear_layer(fc1, 'rnasq_fc2', f2_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)


        f3_width = 1000
        fc3 = linear_layer(fc2, 'rnasq_fc3', f3_width, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        self.f4_out_width = 200
        fc4_out = linear_layer(fc3, "rnasq_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=tf.nn.relu, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        f1_width = 1000
        de_fc1 = linear_layer(x, "rnasq_de_fc1", f1_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        f2_width = 1000
        de_fc2 = linear_layer(de_fc1, "rnasq_de_fc2", f2_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        f3_width = 1000
        de_fc3 = linear_layer(de_fc2, "rnasq_de_fc3", f3_width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=tf.nn.relu)

        de_fc4_out = linear_layer(de_fc3, 'rnasq_de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

class LBEXP_SELU:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for LBEXP_selu")
        self.is_training = is_training
        self.code = 'lbexpselu'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 200, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 150, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        self.f3_out_width = 100
        fc3_out = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+"_fc3_out", self.f3_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        de_fc1 = linear_layer(x, self.code+"_de_fc1", 150, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc2 = linear_layer(de_fc1, self.code+"_de_fc2", 200, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc3_out = linear_layer(de_fc2, self.code+'_de_fc3_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc3_out

class LBEXP_RELU:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for LBEXP_relu")
        self.is_training = is_training
        self.code = 'lbexprelu'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 200, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=None, reuse=reuse)

        fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 150, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=None, reuse=reuse)

        self.f3_out_width = 100
        fc3_out = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+"_fc3_out", self.f3_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc3_out

class AutoEncoder_ECFP_Three_BN:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_three_BN")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        fc1 = batch_normalized_linear_layer(x, "fc1", 500, stddev=.01, wd=weight_decay, test=test)
        fc2 = batch_normalized_linear_layer(fc1, 'fc2', 250, stddev=.01, wd=weight_decay, test=test)
        fc3_out = linear_layer(fc2, "fc3_out", 100, stddev=.01, wd=weight_decay)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        de_fc1 = batch_normalized_linear_layer(x, "de_fc1", 250, stddev=.01, wd=weight_decay, test=test)
        de_fc2 = batch_normalized_linear_layer(de_fc1, "de_fc2", 500, stddev=.01, wd=weight_decay, test=test)
        de_fc3_out = linear_layer(de_fc2, 'de_fc3_out', self.width, stddev=.01, \
                            wd=weight_decay, nonlinearity=intense_sigmoid)

        return de_fc3_out

class AutoEncoder_ECFP_Skinny_BN:
    def __init__(self, numFeatures, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for ECFP_three_skinny_BN")
        self.width = numFeatures
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        fc1 = batch_normalized_linear_layer(x, "fc1", 100, stddev=.01, wd=weight_decay, test=test)
        #fc2 = batch_normalized_linear_layer(fc1, 'fc2', 100, stddev=.01, wd=weight_decay, test=test)
        fc3_out = linear_layer(fc1, "fc3_out", 100, stddev=.01, wd=weight_decay)

        return fc3_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training
        de_fc1 = batch_normalized_linear_layer(x, "de_fc1", 100, stddev=.01, wd=weight_decay, test=test)
        #de_fc2 = batch_normalized_linear_layer(de_fc1, "de_fc2", 100, stddev=.01, wd=weight_decay, test=test)
        de_fc3_out = linear_layer(de_fc1, 'de_fc3_out', self.width, stddev=.01, \
                            wd=weight_decay, nonlinearity=intense_sigmoid)

        return de_fc3_out

class FANG_RELU_DRAGON:
    def __init__(self):
        print("classifier for fang_relu_dragon")
        self.code = 'fang_relu_dragon'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        fc3 = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+'_fc3', 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        return fc3

class FANG_RELU_GENE:
    def __init__(self):
        print("classifier for fang_relu_gene")
        self.code = 'fang_relu_gene'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        fc3 = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+'_fc3', 1000, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)


        return fc3

class TOX_RELU:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("classifier for tox_relu")
        self.is_training = is_training
        self.code = 'toxrelu'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 50, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        #fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 50, stddev='Xav', wd=weight_decay, \
        #                            nonlinearity=tf.nn.relu, reuse=reuse)

        #fc3 = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+'_fc3', 25, stddev='Xav', wd=weight_decay, \
        #                            nonlinearity=tf.nn.relu, reuse=reuse)

        self.f4_out_width = 10
        fc4_out = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+"_fc4_out", self.f4_out_width, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=tf.nn.relu, reuse=reuse)

        return fc4_out

class TOX_RELU_REG:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("classifier for tox_relu_reg")
        self.is_training = is_training
        self.code = 'toxrelu_reg'

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        #fc1 = linear_layer(tf.nn.dropout(x, keep_prob), self.code+"_fc1", 50, stddev='Xav', wd=weight_decay, \
        #                            nonlinearity=tf.nn.relu, reuse=reuse)

        #fc2 = linear_layer(tf.nn.dropout(fc1, keep_prob), self.code+'_fc2', 50, stddev='Xav', wd=weight_decay, \
        #                            nonlinearity=tf.nn.relu, reuse=reuse)

        #fc3 = linear_layer(tf.nn.dropout(fc2, keep_prob), self.code+'_fc3', 25, stddev='Xav', wd=weight_decay, \
        #                            nonlinearity=tf.nn.relu, reuse=reuse)

        #fc4 = linear_layer(tf.nn.dropout(fc3, keep_prob), self.code+"_fc4", 20, stddev='Xav', \
        #                            wd=weight_decay, nonlinearity=tf.nn.relu, reuse=reuse)

        regressionLayer = linear_layer(x, self.code+"_reg_layer", 1, stddev='Xav', wd=weight_decay, \
                                    nonlinearity=tf.nn.relu, reuse=reuse)

        return regressionLayer

class RNASEQ_SELU_big:
    def __init__(self, is_training=True):
        # self.width is how many features make up the input/output
        print("AE for RNASEQ_SELU_big")
        self.is_training = is_training

    # Building the encoder
    def encoder(self, x, keep_prob, weight_decay, reuse=None):
        test = not self.is_training

        self.width = x.get_shape().as_list()[1]

        print("x shape", x.get_shape().as_list())
        print("self.width", self.width)
        fc1 = linear_layer(x, "rnasb_fc1", 5000, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc2 = linear_layer(fc1, 'rnasb_fc2', 2000, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)


        fc3 = linear_layer(fc2, 'rnasb_fc3', 400, stddev='selu', wd=weight_decay, \
                                    nonlinearity=selu, reuse=reuse)

        fc4_out = linear_layer(fc3, "rnasb_fc4_out", 200, stddev='Xav', \
                                    wd=weight_decay, nonlinearity=None, reuse=reuse)

        return fc4_out

    # Building the decoder
    def decoder(self, x, keep_prob, weight_decay):
        test = not self.is_training

        de_fc1 = linear_layer(x, "rnasb_de_fc1", 400, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc2 = linear_layer(de_fc1, "rnasb_de_fc2", 2000, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc3 = linear_layer(de_fc2, "rnasb_de_fc3", 5000, stddev='selu', \
                            wd=weight_decay, nonlinearity=selu)

        de_fc4_out = linear_layer(de_fc3, 'rnasb_de_fc4_out', self.width, stddev='Xav', \
                            wd=weight_decay, nonlinearity=None)

        print("out layer has width", self.width)
        return de_fc4_out

