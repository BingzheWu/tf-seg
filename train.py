import tensorflow as tf
import numpy as np
import config
from data import vocdata

FLAGS = tf.app.flags.FLAGS


def train_op( total_loss,  )

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable = False)
        label = tf.placeholder("float", [None])
        shape = tf.placeholder("int32", [4])
