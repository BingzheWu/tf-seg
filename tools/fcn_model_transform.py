import caffe
import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append('../')
import config
FLAGS = tf.app.flags.FLAGS
vgg_np_model_dir = FLAGS.vgg_np_model_dir
def caffe2npy(model_def, model_weights):
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    for layer_name, param in net.params.iteritems():
        print param.__dict__
        if len(param) == 2:    
            print layer_name +'\t' + str(param[0].data.shape), str(param[1].data.shape)
            np.save(os.path.join(vgg_np_model_dir, layer_name+'_conv'), param[0].data.transpose((2,3,1,0)))
            np.save(os.path.join(vgg_np_model_dir, layer_name+'_bias'), param[1].data)
        else:
            print 11111111
            np.save(os.path.join(vgg_np_model_dir, layer_name), param[0].data.transpose((2,3,1,0)))
            print layer_name  + str(param[0].data.shape)
def main():
    args = sys.argv
    model_def = args[1]
    model_weights = args[2]
    caffe2npy(model_def, model_weights)

if __name__ == '__main__':
    main()
