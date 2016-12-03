#!/usr/bin/env python
# coding=utf-8
import config
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
def loss(logits, labels, head = np.ones([FLAGS.num_classes])):
    '''
    labels's shape is [1, w, h, 1]
    '''
    head =np.array( [0.4] + list(np.ones(20)))
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, FLAGS.num_classes))
        epsilon = tf.constant(value = 1e-10)
        logits = logits + epsilon
        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth = FLAGS.num_classes), (-1, FLAGS.num_classes))
        softmax = tf.nn.softmax(logits)
        print softmax.get_shape()
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax+epsilon), head), reduction_indices = [1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        #regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    return loss
 
