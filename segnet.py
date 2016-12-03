import tensorflow as tf
import config
import tensorflow as tf
slim = tf.contrib.slim
from utils import weight_init
FLAGS = tf.app.flags.FLAGS
def segnet(image, use_cpu = False, scope = 'segnet'):
    if use_cpu:
        device = '/cpu:0'
    else:
        device = '/gpu:0'
    shape = image.get_shape().as_list()
    with tf.variable_scope(scope, 'basic', [image]) as sc:
        with tf.device(device):
            conv1_1 = slim.conv2d(image, 64, [3,3], scope = 'conv1_1', weights_initializer = weight_init('conv1_1_conv.npy'),
                                 biases_initializer = weight_init('conv1_1_bias.npy'))
            conv1_2 = slim.conv2d(conv1_1,64, [3,3], scope = 'conv1_2',  weights_initializer = weight_init('conv1_2_conv.npy'), 
                    biases_initializer = weight_init('conv1_2_bias.npy') ,padding = 'SAME' )
            pool1 = slim.max_pool2d(conv1_2, [2,2], stride = 2,  scope = 'pool1')
            #pool1, pool1_idx = tf.nn.max_pool_with_argmax(conv1_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID') 
            conv2_1 = slim.conv2d(pool1, 128, [3,3], scope = 'conv2_1',  weights_initializer = weight_init('conv2_1_conv.npy'), 
                    biases_initializer = weight_init('conv2_1_bias.npy') ,padding = 'SAME' )
            conv2_2 = slim.conv2d(conv2_1, 128, [3,3], scope = 'conv2_2',  weights_initializer = weight_init('conv2_2_conv.npy'), 
                    biases_initializer = weight_init('conv2_2_bias.npy') ,padding = 'SAME' )
            pool2 = slim.max_pool2d(conv2_2, [2,2], stride = 2,  scope = 'pool2', padding = 'VALID')
            #pool2, pool2_idx = tf.nn.max_pool_with_argmax(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2') 
            
            conv3_1 = slim.conv2d(pool2, 256, [3,3], scope = 'conv3_1',  weights_initializer = weight_init('conv3_1_conv.npy'), 
                    biases_initializer = weight_init('conv3_1_bias.npy') ,padding = 'SAME' )
            conv3_2 = slim.conv2d(conv3_1, 256, [3,3], scope = 'conv3_2', weights_initializer = weight_init('conv3_2_conv.npy'), 
                    biases_initializer = weight_init('conv3_2_bias.npy') ,padding = 'SAME' )
            conv3_3 = slim.conv2d(conv3_2, 256, [3,3], scope = 'conv3_3', weights_initializer = weight_init('conv3_3_conv.npy'), 
                    biases_initializer = weight_init('conv3_3_bias.npy') ,padding = 'SAME' )
            #pool3, pool3_idx = tf.nn.max_pool_with_argmax(conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool3') 
            pool3 = slim.max_pool2d(conv3_3, [2,2], stride = 2,  scope = 'pool3', padding = 'VALID')
            conv4_1 = slim.conv2d(pool3, 512, [3,3], scope = 'conv4_1',weights_initializer = weight_init('conv4_1_conv.npy'), 
                    biases_initializer = weight_init('conv4_1_bias.npy') ,padding = 'SAME' )
            conv4_2 = slim.conv2d(conv4_1, 512, [3,3], scope = 'conv4_2',  weights_initializer = weight_init('conv4_2_conv.npy'), 
                    biases_initializer = weight_init('conv4_2_bias.npy') ,padding = 'SAME' )
            conv4_3 = slim.conv2d(conv4_2, 512, [3,3], scope = 'conv4_3',  weights_initializer = weight_init('conv4_3_conv.npy'), 
                    biases_initializer = weight_init('conv4_3_bias.npy') ,padding = 'SAME' )
            pool4 = slim.max_pool2d(conv4_3,[2,2], scope = 'pool4' )
            #pool4, pool4_idx = tf.nn.max_pool_with_argmax(conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = "pool4") 
            conv5_1 = slim.conv2d(pool4, 512, [3,3], scope = 'conv5_1', weights_initializer = weight_init('conv5_1_conv.npy'), 
                    biases_initializer = weight_init('conv5_1_bias.npy') ,padding = 'SAME' )
            conv5_2 = slim.conv2d(conv5_1, 512, [3,3], scope = 'conv5_2', weights_initializer = weight_init('conv5_2_conv.npy'), 
                    biases_initializer = weight_init('conv5_2_bias.npy') ,padding = 'SAME' )
            conv5_3 = slim.conv2d(conv5_2, 512, [3,3], scope = 'conv5_3',  weights_initializer = weight_init('conv5_3_conv.npy'), 
                    biases_initializer = weight_init('conv5_3_bias.npy') ,padding = 'SAME' )
            #pool5, pool5_idx = tf.nn.max_pool_with_argmax(conv5_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5') 
            pool5 = slim.max_pool2d(conv5_3,[2,2], scope = 'pool5' )
            ###begin decoder
            upsample5 = slim.conv2d_transpose(pool5, 512, [2,2], stride = 2, padding = 'VALID', scope = 'upsample5', activation_fn = None  )
            #conv5_d = slim.conv2d(upsample5, )
            upsample4 = slim.conv2d_transpose(upsample5, 512, [2,2], stride = 2, padding = 'VALID', scope = 'upsample4', activation_fn = None)
            upsample3 = slim.conv2d_transpose(upsample4, 256, [2,2], stride = 2, padding = 'VALID', scope = 'upsample3', activation_fn = None)
            upsample2 = slim.conv2d_transpose(upsample3, 128, [2,2], stride = 2, padding = 'VALID', scope = 'upsample2', activation_fn = None)
            upsample1 = slim.conv2d_transpose(upsample2, 128, [2,2], stride = 2, padding = 'VALID', scope = 'upsample1', activation_fn = None)
            ##classifier
            upsample1 = tf.image.resize_bilinear(upsample2, [shape[1], shape[2]], name = 'resize')
            conv_classifier = slim.conv2d(upsample1, FLAGS.num_classes, [1,1],padding = 'VALID', scope = 'score' )
            logit = conv_classifier
            # shape of logit is [1, w, h, 21]
    return logit
