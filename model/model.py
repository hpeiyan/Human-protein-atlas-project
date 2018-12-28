from keras.models import Sequential
from keras import layers
from keras.applications import VGG19
from keras.applications import ResNet50
from constant import *


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
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def fineTuneVGG19Model(self):
        '''
        事实证明，对于我们的图片，模型表现差劲
        return:
        '''
        model = Sequential()
        vgg = VGG19(include_top=False,
                    weights='imagenet',
                    pooling='max',
                    input_shape=self.input_shape,
                    classes=n_classes)
        for layer in vgg.layers:
            if 'block5' in layer.name:
                layer.trainable = False
        model.add(vgg)
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def fineTuneModel(self):
        model = Sequential()
        res_net = ResNet50(include_top=False,
                           weights='imagenet',
                           input_shape=self.input_shape,
                           classes=n_classes)
        model.add(res_net)
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model
