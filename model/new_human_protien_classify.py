import os
import pandas
from IPython.display import display
import numpy as np
from cutsom_generator import CustomGenerator, CustomImageGenerator
from model import MyModel
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from evaluate import f1
from plot_info import plot
from keras.callbacks import ModelCheckpoint

main_dir = r'/home/bigdata/Documents/DeepLearningProject/datasets/human-protien-data'
train_dir = os.path.join(main_dir, 'train')
test_dir = os.path.join(main_dir, 'test')
info_img_dir = os.path.join(main_dir, 'config.png')
weight_dir = os.path.join(main_dir, 'weights.hdf5')
input_dim = 224
input_channel = 3
batch_size = 128
n_classes = 28
epochs = 50
THRESHOLD = 0.5
train_data_gen_args = dict(rescale=1. / 255,
                           rotation_range=45,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.2,
                           zoom_range=0.2,
                           channel_shift_range=0.2,
                           brightness_range=(0.3, 1.0),
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='constant',
                           cval=0)

val_data_gen_args = dict(rescale=1. / 255)

train_csv = pandas.read_csv(os.path.join(main_dir, 'train.csv'))
display(train_csv.head(5))


def get_train_sample():
    sample_y = np.array([target.split(' ') for target in train_csv['Target']])
    return np.array([Id for Id in train_csv['Id']]), list(zip(train_csv['Id'], sample_y))
    # return np.array([Id for Id in train_csv['Id']]), dict(zip(train_csv['Id'], sample_y))


# def get_test_sample():
#     test_x = []
#     for test_sample in os.listdir(test_dir):
#         if test_sample.endswith("red.png"):
#             test_x.append(test_sample[:-8])
#     return test_x


def get_test_sample():
    return list(set([test_sample[:36] for test_sample in os.listdir(test_dir)]))


sample_x, sample_y = get_train_sample()
X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=0.3, random_state=42)
y_train = dict(y_train)
y_val = dict(y_val)
# y_train = dict(y_train)
# display(sample_x, sample_y)
# test_x = get_test_sample()
# display(test_x)

# generator_train = CustomGenerator(root_path=train_dir,
#                                   sample_x=X_train,
#                                   labels=y_train,
#                                   batch_size=batch_size,
#                                   dim=(input_dim, input_dim),
#                                   n_channels=input_channel,
#                                   n_classes=n_classes,
#                                   shuffle=True)
#
# generator_val = CustomGenerator(root_path=train_dir,
#                                 sample_x=X_val,
#                                 labels=y_val,
#                                 batch_size=batch_size,
#                                 dim=(input_dim, input_dim),
#                                 n_channels=input_channel,
#                                 n_classes=n_classes,
#                                 shuffle=False)

img_train_augment = CustomImageGenerator(**train_data_gen_args)
generator_train = img_train_augment.flow_from_gen(root_path=train_dir,
                                                  sample_x=X_val,
                                                  labels=y_val,
                                                  batch_size=batch_size,
                                                  dim=(input_dim, input_dim),
                                                  n_channels=input_channel,
                                                  n_classes=n_classes,
                                                  shuffle=False)

img_val_augment = CustomImageGenerator(**val_data_gen_args)
generator_val = img_val_augment.flow_from_gen(root_path=train_dir,
                                              sample_x=X_val,
                                              labels=y_val,
                                              batch_size=batch_size,
                                              dim=(input_dim, input_dim),
                                              n_channels=input_channel,
                                              n_classes=n_classes,
                                              shuffle=False)

checkpoint = ModelCheckpoint(weight_dir,
                             monitor='val_f1',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max',
                             period=1)

model = MyModel(input_shape=(input_dim, input_dim, input_channel)).buildModel()
model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc', f1])
history = model.fit_generator(generator=generator_train,
                              steps_per_epoch=len(generator_train),
                              validation_data=generator_val,
                              validation_steps=len(generator_val),
                              epochs=epochs,
                              callbacks=[checkpoint])

plot(history, info_img_dir)
