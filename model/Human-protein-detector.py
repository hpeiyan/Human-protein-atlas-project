import os
import matplotlib.pyplot as plt
from keras import models, layers
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import cv2
from model.Net import CNNNet
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import display
import pickle
import gc
import keras.backend as K
import tensorflow as tf

ids = []
labels = []
train_data_paths = []
train_data = []
test_data_paths = []
test_ids = []
# colors = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
colors = ['_red.png', '_green.png', '_blue.png']
build_new = True

data_dir = r'/home/bigdata/Documents/DeepLearningProject/datasets/human-protien-data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
train_csv = os.path.join(data_dir, 'train.csv')
model_dir = r'./model.hdf5'
detect_result_dir = r'./submission.csv'
THRESHOLD = 0.05

with open(train_csv) as csv:
    content = csv.read()
lines = content.split('\n')
for index, line in enumerate(lines):
    if index == 0:
        continue
    line_data = line.split(',')
    if not len(line_data) == 2:
        continue
    id = line_data[0]
    label_arr = line_data[1]
    ids.append(id)
    labels.append(set(map(int, label_arr.split())))

mlb = MultiLabelBinarizer()
train_y = mlb.fit_transform(labels)

for id in ids:
    train_data_paths.append(os.path.join(train_dir, id))


def f1(y_true, y_pred):
    # y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# def f1(y_true, y_pred):
#     tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
#     fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
#
#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())
#
#     f1 = 2 * p * r / (p + r + K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return K.mean(f1)


test_data_gen = ImageDataGenerator()
data_gen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

model = CNNNet.buildFineTuneNet()
model.compile(optimizer=RMSprop(1e-3),
              loss='binary_crossentropy',
              metrics=['acc', f1])


def trainSetGenerator(data_paths, labels, numFiles):
    x_train = np.zeros(shape=(numFiles, 128, 128, 3))
    y_train = np.zeros(shape=(numFiles, 28))
    full_data = np.zeros(shape=(128, 128, 3))
    count = 0
    for index, path in enumerate(data_paths):
        for i, color in enumerate(colors):
            img_path = path + color
            img_arr = img_to_array(load_img(img_path, target_size=(128, 128), color_mode='grayscale')) / 255.0
            full_data[:, :, i] = img_arr[:, :, 0]
        x_train[count, :, :, :] = full_data
        y_train[count, :] = labels[index]
        count += 1
        if count == numFiles or index == len(data_paths):
            count = 0
            print('yield data: {}'.format(count))
            yield (x_train, y_train)


def buildTrainModel(train_x, label_train):
    X_train, X_test, y_train, y_test = train_test_split(train_x, label_train, shuffle=True, test_size=0.1)

    data_gen.fit(X_train)
    train_gen = data_gen.flow(x=X_train, y=y_train, batch_size=32)
    test_gen = test_data_gen.flow(x=X_test, y=y_test, batch_size=32)
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_gen),
                                  validation_steps=len(test_gen),
                                  validation_data=test_gen)


def testSetGetter(test_data_paths):
    x_test = np.zeros(shape=(len(test_data_paths), 128, 128, 3))
    ceil_data = np.zeros(shape=(128, 128, 3))
    for index, path in enumerate(test_data_paths):
        for i, color in enumerate(colors):
            img_arr = img_to_array(load_img(path + color, target_size=(128, 128), color_mode='grayscale')) / 255.0
            ceil_data[:, :, i] = img_arr[:, :, 0]
        x_test[index, :, :, :] = ceil_data

    return x_test


def operateTest():
    for file in os.listdir(test_dir):
        if file.endswith('_green.png'):
            test_ids.append(file[:-10])
            test_data_paths.append(os.path.join(test_dir, file[:-10]))
    test_x = testSetGetter(test_data_paths)
    model = models.load_model(model_dir, custom_objects={'f1': f1})
    prediction = model.predict(x=test_x,
                               batch_size=128)

    target = [' '.join(map(str, x)) for x in mlb.inverse_transform(np.round(prediction))]

    output = pd.DataFrame(data=list(zip(test_ids, target)),
                          columns=['Id', 'Predicted'])
    output.to_csv(detect_result_dir)
    print('Test Done')
    result = pd.read_csv('./submission.csv')
    display(result)


if build_new:
    # train
    for epoch in range(50):
        for (x_train, y_train) in trainSetGenerator(train_data_paths, train_y, 8000):
            buildTrainModel(x_train, y_train)

    model.save(model_dir)
    print('epoch train done')
else:
    # test
    operateTest()
