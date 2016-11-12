import tensorflow as tf
import tensorflow.contrib.slim as slim
from voc_label import labelcolormap
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from utils import weight_init

def fcn8(images, mean = None, num_classes = 21,  scope = 'fcn_8s'):
    '''
    Args:
    images: 4-d input tensor
    mean: the mean values of pixels 
    num_classes: num of classes 
    '''
    with tf.variable_scope(scope, 'fcn8s', [images]) as sc:
        if mean:
            images = tf.sub(images, mean)
        pad_images = tf.pad(images, [[0,0], [100, 100], [100, 100], [0,0]])
        conv1_1 = slim.conv2d(pad_images,64, [3,3], scope = 'conv1_1',  weights_initializer = weight_init('conv1_1_conv.npy'), 
                biases_initializer = weight_init('conv1_1_bias.npy') ,padding = 'VALID' )
        print conv1_1.get_shape()
        conv1_2 = slim.conv2d(conv1_1,64, [3,3], scope = 'conv1_2',  weights_initializer = weight_init('conv1_2_conv.npy'), 
                biases_initializer = weight_init('conv1_2_bias.npy') ,padding = 'SAME' )
        if conv1_2.get_shape().as_list()[1] % 2 == 1:
            conv1_2 = tf.pad(conv1_2, [[0,0],[0,1],[0,1],[0,0]])
        print conv1_2.get_shape()
        pool1 = slim.max_pool2d(conv1_2, [2,2], stride = 2,  scope = 'pool1')
        print pool1.get_shape()
        conv2_1 = slim.conv2d(pool1, 128, [3,3], scope = 'conv2_1',  weights_initializer = weight_init('conv2_1_conv.npy'), 
                biases_initializer = weight_init('conv2_1_bias.npy') ,padding = 'SAME' )
        conv2_2 = slim.conv2d(conv2_1, 128, [3,3], scope = 'conv2_2',  weights_initializer = weight_init('conv2_2_conv.npy'), 
                biases_initializer = weight_init('conv2_2_bias.npy') ,padding = 'SAME' )
        print conv2_2.get_shape()
        if conv2_2.get_shape().as_list()[1]%2 == 1:
            conv2_2 = tf.pad(conv2_2, [[0,0], [0,1], [0,1], [0,0]])
        pool2 = slim.max_pool2d(conv2_2, [2,2], stride = 2,  scope = 'pool2', padding = 'VALID')
        print pool2.get_shape()
        conv3_1 = slim.conv2d(pool2, 256, [3,3], scope = 'conv3_1',  weights_initializer = weight_init('conv3_1_conv.npy'), 
                biases_initializer = weight_init('conv3_1_bias.npy') ,padding = 'SAME' )
        conv3_2 = slim.conv2d(conv3_1, 256, [3,3], scope = 'conv3_2', weights_initializer = weight_init('conv3_2_conv.npy'), 
                biases_initializer = weight_init('conv3_2_bias.npy') ,padding = 'SAME' )
        conv3_3 = slim.conv2d(conv3_2, 256, [3,3], scope = 'conv3_3', weights_initializer = weight_init('conv3_3_conv.npy'), 
                biases_initializer = weight_init('conv3_3_bias.npy') ,padding = 'SAME' )
        if conv3_3.get_shape().as_list()[1] % 2 == 1:
            conv3_3 = tf.pad(conv3_3, [[0,0], [0,1], [0,1], [0,0]])
        pool3 = slim.max_pool2d(conv3_3, [2,2], stride = 2, scope = 'pool3')
        print pool3.get_shape()
        conv4_1 = slim.conv2d(pool3, 512, [3,3], scope = 'conv4_1',weights_initializer = weight_init('conv4_1_conv.npy'), 
                biases_initializer = weight_init('conv4_1_bias.npy') ,padding = 'SAME' )
        conv4_2 = slim.conv2d(conv4_1, 512, [3,3], scope = 'conv4_2',  weights_initializer = weight_init('conv4_2_conv.npy'), 
                biases_initializer = weight_init('conv4_2_bias.npy') ,padding = 'SAME' )
        conv4_3 = slim.conv2d(conv4_2, 512, [3,3], scope = 'conv4_3',  weights_initializer = weight_init('conv4_3_conv.npy'), 
                biases_initializer = weight_init('conv4_3_bias.npy') ,padding = 'SAME' )
        if conv4_3.get_shape().as_list()[1] % 2 == 1:
            conv4_3 = tf.pad(conv4_3, [[0,0], [0,1], [0,1], [0,0]])
        pool4 = slim.max_pool2d(conv4_3,[2,2], scope = 'pool4' )
        print pool4.get_shape()
        conv5_1 = slim.conv2d(pool4, 512, [3,3], scope = 'conv5_1', weights_initializer = weight_init('conv5_1_conv.npy'), 
                biases_initializer = weight_init('conv5_1_bias.npy') ,padding = 'SAME' )
        conv5_2 = slim.conv2d(conv5_1, 512, [3,3], scope = 'conv5_2', weights_initializer = weight_init('conv5_2_conv.npy'), 
                biases_initializer = weight_init('conv5_2_bias.npy') ,padding = 'SAME' )
        conv5_3 = slim.conv2d(conv5_2, 512, [3,3], scope = 'conv5_3',  weights_initializer = weight_init('conv5_3_conv.npy'), 
                biases_initializer = weight_init('conv5_3_bias.npy') ,padding = 'SAME' )
        if conv5_3.get_shape().as_list()[1] % 2 == 1:
            conv5_3 = tf.pad(conv5_3, [[0,0], [0,1], [0,1], [0,0]])
        pool5 = slim.max_pool2d(conv5_3, [2,2], scope = 'pool5')
        fc6 = slim.conv2d(pool5, 4096, [7,7], scope = 'fc6', padding = 'VALID', weights_initializer = weight_init('fc6_conv.npy'),
                biases_initializer = weight_init('fc6_bias.npy'))
        fc7 = slim.conv2d(fc6, 4096, [1,1], scope = 'fc7', padding = 'VALID', weights_initializer = weight_init('fc7_conv.npy'),
                biases_initializer = weight_init('fc7_bias.npy'))
        score_fr = slim.conv2d(fc7, num_classes, [1,1], padding = 'VALID', scope = 'score_fr', 
                weights_initializer = weight_init('score_fr_conv.npy'), biases_initializer = weight_init('score_fr_bias.npy'), activation_fn = None)
        upscore2 = slim.conv2d_transpose(score_fr, num_classes, [4,4], stride = 1, scope = 'upscore2', weights_initializer = weight_init('upscore2.npy'), activation_fn = None)
        score_pool4 = slim.conv2d(pool4, num_classes, [1,1], padding = 'VALID', scope = 'score_pool4',
                weights_initializer = weight_init('score_pool4_conv.npy'), biases_initializer = weight_init('score_pool4_bias.npy'), activation_fn = None )
        score_pool4c = tf.image.resize_bilinear(score_pool4, upscore2.get_shape().as_list()[1:3])
        fuse_pool4 = tf.add(score_pool4c, upscore2)
        upscore_pool4 = slim.conv2d_transpose(fuse_pool4, num_classes, [4,4], stride = 2, scope = 'upscore_pool4', weights_initializer = weight_init('upscore_pool4.npy'), activation_fn = None)
        score_pool3 = slim.conv2d(pool3, num_classes, [1,1], padding = 'VALID', scope = 'score_pool3', activation_fn = None)
        score_pool3c = tf.image.resize_bilinear(score_pool3, upscore_pool4.get_shape().as_list()[1:3])
        fuse_pool3 = tf.add(upscore_pool4, score_pool3c)
        upscore8 = slim.conv2d_transpose(fuse_pool3, num_classes, [16, 16], stride = 8, scope = 'upscore8', weights_initializer = weight_init('upscore8.npy'), activation_fn = None)
        score = tf.image.resize_bilinear(upscore8, images.get_shape().as_list()[1:3])
        return upscore2, score
def main():
    images = tf.placeholder("float",[None,329,329,3])
    im = misc.imread('data/1.jpg')
    im = misc.imresize(im,(329, 329))
    im = im - np.array((104.00698793,116.66876762,122.67891434))
    im = np.expand_dims(im, axis = 0)
    pool5, infer = fcn8(images)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    pool5, result = sess.run([pool5, infer], feed_dict = {images:im})
    out = result.argmax(axis =3)
    color_bar = labelcolormap(21)
    result_list = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            result_list.append(color_bar[out[i][j]])
    result_list = np.array(result_list)
    result_list = result_list.reshape((out.shape[1],out.shape[2],3))
    print pool5
    plt.imshow(result_list)
    plt.show()
def debug():
    im = misc.imread('data/1.jpg')
    im = misc.imresize(im, (329, 329))
    im = im - np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.expand_dims(im, axis=0)
    with tf.Graph().as_default():
        images = tf.placeholder("float", [None, 329, 329, 3])
        with tf.Session() as sess:
            pool5, infer = fcn8(images)
            sess.run(tf.initialize_all_variables())
            sess.run(infer, feed_dict={images:im})
            for op in sess.graph.get_operations():
                print op.name
if __name__ == '__main__':
    main()

