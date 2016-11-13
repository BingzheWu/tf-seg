import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("vgg_np_model_dir", '/home/bingzhe/data/models/segmentation/fcn8s/npy/','the path to vgg numpy model')
tf.app.flags.DEFINE_string("voc_dir", '/opt/dataset/VOCdevkit/VOC2012/', "the dir path to VOC dataset")

