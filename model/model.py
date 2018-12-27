from keras.models import Sequential
from keras import layers


class MyModel():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def buildModel(self):
        model = Sequential()
        model.add(layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model
