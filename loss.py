import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models import fcn8

FLAGS = tf.app.flags.FLAGS
def loss(logits, loss):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
    total_loss = regularization_loss + cross_entropy_mean
    return total_loss

