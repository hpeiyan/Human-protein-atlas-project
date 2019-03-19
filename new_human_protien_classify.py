import pandas
from IPython.display import display
import numpy as np
from cutsom_generator import CustomGenerator
from model import MyModel
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from evaluate import f1
from evaluate import Metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from plot_info import plot
from constant import *
import matplotlib
import matplotlib.pyplot as plt
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.switch_backend('Agg')
# matplotlib.use('agg')
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

X_train, X_val, y_train, y_val = train_test_split(sample_x, sample_y, test_size=0.2,
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
                             monitor='val_f1',
                             mode='max',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             period=1)
metrics = Metrics()

ear_stop = EarlyStopping(patience=patience)
myModel = MyModel(input_shape=(input_dim, input_dim, input_channel))
model = myModel.DenseNet()
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=[f1])
history = model.fit_generator(generator=generator_train,
                              steps_per_epoch=len(generator_train),
                              validation_data=generator_val,
                              validation_steps=len(generator_val),
                              epochs=epochs,
                              callbacks=[checkpoint, ear_stop])
# callbacks=[metrics])

# plot(history,info_img_dir)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_f1 = history.history['f1']
val_f1 = history.history['val_f1']

epochs = range(1, len(val_f1) + 1)

plt.plot(epochs, train_f1, label='train_f1')
plt.plot(epochs, val_f1, label='val_f1')
plt.legend()

plt.figure()
plt.plot(epochs, train_loss, label='train_loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.legend()
plt.show()
plt.savefig("image.png")
