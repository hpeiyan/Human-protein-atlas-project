from keras.models import Sequential
from keras import layers
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionResNetV2
from keras.applications import DenseNet169
from keras.layers import Input, BatchNormalization, Conv2D
from keras.models import Model
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
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def DenseNet(self):
        model = Sequential()
        dn = DenseNet169(include_top=False,
                         weights=None,
                         input_shape=self.input_shape,
                         classes=n_classes)
        dn.trainable = True
        dn.summary()
        model.add(dn)
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def ftDenseNet(self):
        model = Sequential()
        dn = DenseNet169(include_top=False,
                         weights='imagenet',
                         input_shape=self.input_shape,
                         classes=n_classes)
        dn.trainable = True
        for l in dn.layers:
            if not l.name.startswith('conv5'):
                l.trainable = False
        l = dn.layers[-1]
        print(l.name)
        l.trainable = True
        dn.summary()
        model.add(dn)
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def fineTuneVGG19Model(self):
        model = Sequential()
        vgg = VGG19(include_top=False,
                    # weights='imagenet',
                    weights=None,
                    pooling='max',
                    input_shape=self.input_shape,
                    classes=n_classes)
        vgg.trainable = True
        # for layer in vgg.layers:
        #     if not (layer.name.startswith('block5') or layer.name.startswith('block4')):
        #         layer.trainable = False
        vgg.summary()
        model.add(vgg)
        # model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='sigmoid'))
        model.summary()
        return model

    def ResNetModel(self):
        '''
        without weights
        :return:
        '''
        model = Sequential()
        res_net = ResNet50(include_top=False,
                           weights=None,
                           input_shape=self.input_shape,
                           classes=n_classes)
        res_net.summary()
        model.add(res_net)
        model.add(layers.Flatten())
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

    def fine_tune_inception(self):
        pretrain_model = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape)

        input_tensor = Input(shape=self.input_shape)
        bn = BatchNormalization()(input_tensor)
        x = pretrain_model(bn)
        x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(n_classes, activation='sigmoid')(x)
        model = Model(input_tensor, output)

        return model
