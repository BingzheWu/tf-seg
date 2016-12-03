import tensorflow as tf
import os
import numpy as np
import config
from data import vocdata
from segnet import segnet
FLAGS = tf.app.flags.FLAGS
from loss_ import loss

batch_size = 1
def train_op(loss, global_step):
    batch_size = 1
    total_samples = 2020
    num_batches_per_epoch = 2020
    lr = 0.001
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name = 'train')
    return train_op

def train():
    checkpoint_dir = "/home/bingzhe/data/models/segmentation/seg_net/"
    image_pl = tf.placeholder("float", shape = [batch_size, 224, 224, 3])
    label_pl = tf.placeholder(tf.int32, shape = [batch_size, 224, 224, 1])
    logit = segnet(image_pl)
    total_loss = loss(logit, label_pl)
    global_step = tf.Variable(0, trainable = False)
    train_op_ = train_op(total_loss, global_step = global_step)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for epoch in range(20):
            for idx in range(2000):
                image, label = vocdata().load_one_image(idx)
                feed_dict = {image_pl: image, label_pl: label}
                _, loss_ = sess.run([train_op_, total_loss], feed_dict)
                if idx % 10 ==0:
                    print loss_
                if idx % 1000 ==0:
                    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = idx+epoch*2000)
if __name__ == '__main__':
    train()




