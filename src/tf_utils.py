"""
Contains training and testing utility functions
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import re
import numpy as np


# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def check_bad_values(*loss_values):
    nanned = any(np.isnan(l) for l in loss_values)
    assert not nanned, 'Model diverged with loss = NaN'

def remove_ind(arr, i):
    """
    returns new list with arr[i] removed
    """
    return arr[:i] + arr[i+1:]


def squared_difference(x, y, name=None):
    return tf.square(x - y, name=name)


def split_tensor_to_list(tensor):
    """
    splits a tensor of size (splits, A, B, C, ... )
    into a splits-length list of tensors of shape (A, B, C, ... )
    """
    splits = tensor.get_shape().as_list()[0]
    return [tf.squeeze(t, [0]) for t in tf.split(0, splits, tensor)]


def whiten_image_batch_tensor(tensor):
    """
    Takes in a tensor of size (batch_size, height, width, channels)
    and whiten it so each image has zero mean and unit norm
    """
    with tf.variable_scope("whitening") as scope:
        tensor_shape = tensor.get_shape().as_list()
        batch_size = tensor_shape[0]
        num_elements = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        mean, var = tf.nn.moments(tensor, axes=[1, 2, 3])
        stddev = tf.sqrt(var)
        adj_stddev = tf.maximum(stddev, 1.0 / (num_elements)**.5)
        whitened = tf.div(
            tensor - tf.reshape(mean, [batch_size, 1, 1, 1]),
            tf.reshape(adj_stddev, [batch_size, 1, 1, 1]),
            name="{}_whitened".format(tensor.name)
        )
    return whitened


def get_saver(moving_average_decay, nontrainable_restore_names=None):
    """
    Gets the saver that restores the variavles for testing

    by default, restores to moving exponential average versions of
    trainable variables

    if nontrainable_restore_names is set, then restores
    nontrainable variables that match (this can be used
    for restoring mean averages for batch normalization)
    """
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay
    )
    variables_to_restore = {}
    for v in tf.global_variables():
        # if the variable is trainable or its name has the desird substring
        if v in tf.trainable_variables() or nontrainable_restore_names is not None and nontrainable_restore_names in v.name:
            print(v.name)
            restore_name = variable_averages.average_name(v)
        else:
            restore_name = v.op.name
        variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)
    return saver


def microsoft_initilization_std(shape):
    """
    Convolution layer initialization as described in:
    http://arxiv.org/pdf/1502.01852v1.pdf
    """
    if len(shape) == 4:
        n = shape[0] * shape[1] * shape[3]
        return (2.0 / n)**.5
    elif len(shape) == 2:
        return (2.0 / shape[1])**.5
    else:
        assert False, "Only works on normal layers"

def xavier_initialization_std(shape):
    """
    xavier glorot init
    """
    std = (2.0 / (shape[1] + shape[0])) ** .5

    return std

def selu_initialization_std(shape):
    """
    initialization meant to be used in conjunction with selu non-linearity. sqrt(1/fan_in)
    """
    std = (1.0/shape[0]) ** .5
    print("selu init std", shape, std)
    return std

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    activiation_mean, activation_variance = tf.nn.moments(x, axes=[0])
    tf.summary.histogram(tensor_name + '/activation_mean', activiation_mean)
    tf.summary.histogram(tensor_name + '/activation_var', activation_variance)
    tf.summary.scalar(tensor_name + '/activation_var_mean', tf.reduce_mean(activation_variance))


def add_summary(x):
    tf.summary.histogram(x.op.name+"/hist", x)

def moment_summary(x, name):
    x_mu, x_var = tf.nn.moments(x, axes=[0])
    print(x_var.get_shape().as_list(), "VAR")
    tf.summary.scalar("Means/{}".format(name), tf.reduce_mean(x_mu))
    tf.summary.scalar("Variances/{}".format(name), tf.reduce_mean(x_var))


def _create_variable(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _variable_with_weight_decay(name, shape, wd, stddev="MSFT"):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    if stddev == "MSFT":
        # use microsoft initialization
        stddev = microsoft_initilization_std(shape)
    elif stddev == "Xav":
        stddev = xavier_initialization_std(shape)
    elif stddev == "selu":
        stddev = selu_initialization_std(shape)

    var = _create_variable(
        name, shape,
        tf.truncated_normal_initializer(stddev=stddev)
    )
    if wd:
        #print("weight decay is", wd)
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_loss_summaries(main_losses):
    """Add summaries for losses in the model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        main_losses: list of losses that will be added
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + main_losses)

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + main_losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op