# coding: utf-8

import tensorflow as tf

def calc_center_loss(features, labels, alpha):
    # calculate the average value of encoded features to see if they're declining
    avg_val = tf.reduce_mean(features)
    tf.summary.scalar("average activation value", avg_val)

    # calculate max and min
    #maxs = tf.reduce_max(features, axis=0)
    #print("maxs dimensions", maxs.get_shape().as_list())
    #tf.summary.histogram("dimensional max value", maxs)
    #mins = tf.reduce_min(features, axis=0)
    #print("mins dimensions", mins.get_shape().as_list())
    #tf.summary.histogram("dimensional min value", mins)

    num_classes = labels.get_shape().as_list()[1]
    not_one_hot = tf.argmax(labels, axis=1)
    print("not_one_hot", not_one_hot.get_shape().as_list())
    return get_center_loss(features, not_one_hot, alpha, num_classes)


def get_center_loss(unnormalized_features, labels, alpha, num_classes):
    """
    Arguments:
        features: Tensor, characterization of the sample, the general use of a fc layer output, shape should be [batch_size, feature_length].
        labels: Tensor, characterize sample label, not one-hot encoding, shape should be [batch_size].
        num_classes: Integer, indicating how many categories there are in total, how many neurons are output from the network classifier

    Return：
        loss: l2_loss of features-centers
        centers: Tensor, one center per class
        centers_update_op: op used to update centers
    """
    len_features = unnormalized_features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    print("centers shape", centers.get_shape().as_list())

    with tf.variable_scope("center_loss") as scope:
        # normalize features to prevent lazy solution of shriking latent space
        features = tf.nn.l2_normalize(unnormalized_features, dim=0)

        # turns labels into one long list. length of batch_size*num_classes
        labels = tf.reshape(labels, [-1])

        # tf.gather distributes the centers into a batch_size length array using labels
        centers_batch = tf.gather(centers, labels)

        # use average over batch size
        #loss = tf.reduce_mean(centers)
        #loss = tf.nn.l2_loss(features - centers_batch)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - centers_batch), 1))

        # difference between each sample and it's corresponding center
        diff = centers_batch - features

        # update centers with alpha*diff
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast(1 + appear_times, tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

def calc_center_distances(unnormalized_features, labels):
    # calculate all pair wise distance between centers
    # using squared_diff code from https://towardsdatascience.com/stupid-tensorflow-tricks-3a837194b7a0
    with tf.variable_scope("center_penalty") as scope:
        # normalize the features
        features = tf.nn.l2_normalize(unnormalized_features, dim=0)

        # make labels into integer representation
        num_classes = labels.get_shape().as_list()[1]
        not_one_hot = tf.argmax(labels, axis=1)
        labels = tf.reshape(not_one_hot, [-1])

        # make variable to store centers
        # code from: https://learningtensorflow.com/lesson6/
        partitions = tf.dynamic_partition(features, tf.cast(labels, tf.int32), num_classes)
        centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

        # Yaru code for calculating distance matrix
        diff = tf.subtract(tf.expand_dims(centroids,2), tf.expand_dims(tf.transpose(centroids), 0))
        squared = tf.square(diff)
        dist_mat = tf.reduce_sum(squared,1)
        num_pairs = ((num_classes*num_classes)-num_classes) / 2.
        print("pairs of centers", num_pairs)
        total = tf.divide(tf.reduce_sum(dist_mat), num_pairs)

        return total

