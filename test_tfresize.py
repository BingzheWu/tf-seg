import tensorflow as tf
from scipy import misc
import numpy as np

def resize_op(image, size = [96,96]):
    return tf.image.resize_bilinear(image, size)

def test(image_name):
    img = misc.imread(image_name)
    img = np.expand_dims(img, axis = 0)
    img = tf.constant(img)
    #img = tf.expand_dims(img, 0)
    with tf.Session() as sess:
        op = resize_op(img)
        ans = sess.run(op)
    print ans
if __name__ =='__main__':
    test('data/1.jpg')