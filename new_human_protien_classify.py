import os
import pandas
from IPython.display import display
import numpy as np
from cutsom_generator import CustomGenerator
from model import MyModel
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from evaluate import focal_loss
from plot_info import plot, plot_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from constant import *
from keras.models import load_model
import tqdm
from sklearn.metrics import f1_score as off1
import pickle

print(log_info + 'Apart data mode!!!' if debug_mode else 'Full data mode!!!')

train_csv = pandas.read_csv(os.path.join(main_dir, 'train.csv'))
display(train_csv.head(5))


def get_train_sample():
    sample_y = np.array([target.split(' ') for target in train_csv['Target']])
    return np.array([Id for Id in train_csv['Id']]), list(zip(train_csv['Id'], sample_y))


sample_x, sample_y = get_train_sample()
if debug_mode:
    epochs = 2
    sample_x = sample_x[0:test_samples]
    sample_y = sample_y[0:test_samples]

X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=0.15,
                                                  random_state=42)
y_train = dict(y_train)
y_val = dict(y_val)

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
                             # monitor='val_loss',
                             # mode='min',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             period=1)
ear_stop = EarlyStopping(patience=patience)
myModel = MyModel(input_shape=(input_dim, input_dim, input_channel))
model = myModel.buildModel()
model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc', focal_loss])
history = model.fit_generator(generator=generator_train,
                              steps_per_epoch=len(generator_train),
                              validation_data=generator_val,
                              validation_steps=len(generator_val),
                              epochs=epochs,
                              callbacks=[checkpoint, ear_stop])

# plot(history, info_img_dir)

# local_model = load_model(weight_dir, custom_objects={'f1': f1})
#
# caculate_final_val(generator_val, local_model)
