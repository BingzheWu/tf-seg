import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("vgg_np_model_dir", '/home/bingzhe/data/models/segmentation/fcn8s/npy/','the path to vgg numpy model')
tf.app.flags.DEFINE_string("voc_dir", '/opt/dataset/VOCdevkit/VOC2012/', "the dir path to VOC dataset")
tf.app.flags.DEFINE_integer("num_classes", 21, "the num of labels to predict")
