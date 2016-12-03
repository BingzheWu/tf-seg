import tensorflow as tf
import config
from segnet import segnet
from voc_label  import labelcolormap
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
ckpt_dir = "/home/bingzhe/data/models/segmentation/seg_net/"
def load_image(im_name):
    im = misc.imread(im_name)
    im = misc.imresize(im, (224, 224))
    im = im -np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.expand_dims(im, axis = 0)
    return im
    
def main():
    image_pl = tf.placeholder(tf.float32, shape = (1, 224, 224, 3))
    im = load_image('data/2.jpg')
    score = segnet(image_pl)
    score = tf.nn.softmax(score)
    sess = tf.Session() 
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_dir+'model.ckpt-1000')
    #sess.run(tf.initialize_all_variables())
    score = sess.run(score, feed_dict = {image_pl: im})
    out = score.argmax(axis =3)
    color_bar = labelcolormap(21)
    result_list = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            result_list.append(color_bar[out[i][j]])
    result_list = np.array(result_list)
    result_list = result_list.reshape((out.shape[1],out.shape[2],3))
    plt.imshow(result_list)
    plt.show()
if __name__ == '__main__':
    main()
