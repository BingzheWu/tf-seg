import sys
sys.path.append('../')
import os
import tensorflow as tf
import config
import numpy as np
FLAGS = tf.app.flags.FLAGS
vgg_model_dir = FLAGS.vgg_np_model_dir
def weight_init(conv_name):
    w = np.load(os.path.join(vgg_model_dir,str(conv_name)))
    #w = np.transpose(w, (2,3,0,1))
    return tf.constant_initializer(w, dtype = tf.float32)
def get_weight(conv_name):
    w = np.load(os.path.join(vgg_model_dir, str(conv_name)))
    return tf.constant(w)
def cal_shape(in_shape, op_shape, stride,  padding_type):
    '''
    calculate the output shape after operation
    op_shape is a 4-D tensor_shape
    '''
    if padding == 'Same':
        out_shape = in_shape / stride
    else:
        out_shape = in_shape
    return in_shape
if __name__ == '__main__':
    s = weight_init('conv3_3_conv.npy')
