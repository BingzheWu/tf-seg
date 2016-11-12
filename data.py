import tensorflow as tf
import config
import numpy as np
FLAGS = tf.app.flags.FLAGS
from PIL import Image

class VocData():
    def __init__(self, flist_name,rgb_mean,voc_dir = FLAGS.voc_dir):
        self.voc_dir = voc_dir
        self.flist_name = os.path.join(self.voc_dir, flist_name)
        self.mean = np.array(rgb_mean)
        self.f = open(self.flist_name, 'r')
    def _read(self):
        _, data_img_name, label_img_name = self.f.readline().strip('\n').split('\t')
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_img(data_img_name, label_img_name)
        return list(data.items()), list(label.items())
    def _read_img(self, img_name, label_name):
        img = Image.open(os.path.join(self.voc_dir, img_name))
        label = Image.open(os.path.join(self.voc_dir, label_name))
        assert img.size == label.size
        if self.cut_off_size is not None:
            max_hw = max(img.shape[0], img.shape[1])
            min_hw = min(img.shape[0], img.shape[1])
            if min_hw > self.cut_off_size:
                rand_start_max = round(np.random.uniform(0, max_hw - self.cut_off_size - 1))
                rand_start_min = round(np.random.uniform(0, min_hw - self.cut_off_size - 1))
                if img.shape[0] == max_hw:
                    img = img[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min: rand_start_min + self.cut_off_size]
                    label = label[rand_start_max: rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
                else:
                    img = img[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max: rand_start_max + self.cut_off_size]
                    label = label[rand_start_min: rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
            elif max_hw > self.cut_off_size:
                rand_start = round(np.random.uniform(0, max_hw - min_hw -1))
                if img.shape[0] == max_hw:
                    img = img[rand_start_min : rand_start_min + self.cut_off_size]
                    label = label[rand_start_min : rand_start_min + self.cut_off_size]
                else:
                    img = img[:, rand_start : rand_start + min_hw]
                    label = label[:, rand_start:rand_start + min_hw]
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean
        img = np.expand_dims(img, axis = 0)
        label = np.array(label)
        label = np.expand_dims(label, axis = 0)
        return (img, label)
    def provide_data(self):
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]
    def provide_label(self):
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]



