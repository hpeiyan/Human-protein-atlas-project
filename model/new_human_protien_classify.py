import os
import pandas
from IPython.display import display
import numpy as np
from cutsom_generator import CustomGenerator
from model import MyModel
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from evaluate import f1
from plot_info import plot, plot_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from constant import *

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


sample_x, sample_y = get_train_sample()
X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=0.3, random_state=42)
y_train = dict(y_train)
y_val = dict(y_val)
# y_train = dict(y_train)
# display(sample_x, sample_y)
# test_x = get_test_sample()
# display(test_x)

generator_train = CustomGenerator(root_path=train_dir,
                                  sample_x=X_train,
                                  labels=y_train,
                                  batch_size=batch_size,
                                  dim=(input_dim, input_dim),
                                  n_channels=input_channel,
                                  n_classes=n_classes,
                                  augment=True,
                                  shuffle=True)

generator_val = CustomGenerator(root_path=train_dir,
                                sample_x=X_val,
                                labels=y_val,
                                batch_size=batch_size,
                                augment=False,
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
ear_stop = EarlyStopping(monitor='val_f1', mode='max', patience=2)

model = MyModel(input_shape=(input_dim, input_dim, input_channel)).buildModel()
model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc', f1])
history = model.fit_generator(generator=generator_train,
                              steps_per_epoch=len(generator_train),
                              validation_data=generator_val,
                              validation_steps=len(generator_val),
                              epochs=epochs,
                              callbacks=[checkpoint, ear_stop])

plot(history, info_img_dir)
