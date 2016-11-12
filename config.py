import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("vgg_np_model_dir", '/home/ceca/bingzhe/seg_models/caffe/fcn8s','the path to vgg numpy model')
tf.app.flags.DEFINE_string("voc_dir", '/home/ceca/bingzhe/data/VOC2012/', "the dir path to VOC dataset")

