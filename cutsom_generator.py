from keras.utils.data_utils import Sequence
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
import imgaug as ia
# from IPython.display import display
from imgaug import augmenters as iaa
# from model.plot_info import plot_img
# from model.constant import *
from plot_info import plot_img
from constant import *
import cv2


class CustomGenerator(Sequence):

    def __init__(self, root_path, sample_x, labels, batch_size, dim, n_channels=1,
                 n_classes=10, shuffle=True, augment=True):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.sample_x = sample_x
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.root_path = root_path
        self.augment = augment
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, samples_temp, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.zeros((self.batch_size, 28))

        seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        # Generate data
        for i, ID in enumerate(samples_temp):
            # Store sample
            sample_channels = [os.path.join(self.root_path, ID + '_' + color + '.png') for color in colors]
            # np.array([img_to_array(load_img(each_channel)) for each_channel in sample_channels])

            for each, each_channel in enumerate(sample_channels):
                img = img_to_array(load_img(each_channel, target_size=self.dim, color_mode='grayscale'))
                real_img = img[:, :, 0]
                X[i, :, :, each] = real_img

            if self.labels is not None:
                for key in self.labels[ID]:
                    Y[i, int(key)] = 1

        if plot_img:
            print(log_info + "start plotting " + str(index) + "images")
            plot_img(X, os.path.join(merge_img_dir, str(index) + '.png'))
            print(log_info + "ending plot " + str(index) + "images")

        if self.augment:
            X = seq.augment_images(X)

        if self.labels is not None:
            return X / 255.0, Y
        else:
            return X / 255.0

    def __getitem__(self, index):
        indexs = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        sample_temp = [self.sample_x[i] for i in indexs]
        return self.__data_generation(sample_temp, index)

    def __len__(self):
        return np.ceil(len(self.sample_x) / self.batch_size).astype(np.int64)
