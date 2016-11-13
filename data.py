import tensorflow as tf
import config
import numpy as np
import os
FLAGS = tf.app.flags.FLAGS
from PIL import Image
from scipy import misc
class VocData():
    '''
    This is the data class for mxnet
    '''
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
class vocdata(): 
    def __init__(self, voc_dir = FLAGS.voc_dir):
        self.voc_dir = voc_dir
        self.label_dir = os.path.join(self.voc_dir, 'SegmentationClass')
        self.imgs_dir = os.path.join(self.voc_dir, 'JPEGImages')
        #self.filelist = filelist
        #self.mean = rgb_mean
        self.f = self.make_file_list()
    def make_file_list(self):
        image_list = os.listdir(self.label_dir)
        return image_list
    def transform_label(self, label):
        ans = np.zeros(label.shape)
        for i in range(label.shape[0]):
            if label[i] == 255:
                ans[i] = 0
            else:
                ans[i] = label[i]
        return ans
    def load_one_image(self, img_idx):
        '''
        load one image for the reson that voc images have various size
        '''
        img_name = self.f[img_idx].split('.')[0]
        img_path = os.path.join(self.imgs_dir, img_name+'.jpg')
        label_path = os.path.join(self.label_dir, img_name+'.png')
        img = Image.open(img_path)
        label = Image.open(label_path)
        img = np.array(img)
        label = np.array(label)
        label = label.ravel()
        label = self.transform_label(label)
        return img, label
if __name__ == '__main__':
    voc = vocdata()
    image, label = voc.load_one_image(0)
    misc.imshow(image) 




