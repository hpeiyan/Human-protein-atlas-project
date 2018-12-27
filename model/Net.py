from keras import models, layers
from keras.applications import VGG16
from keras.regularizers import l1_l2
from keras import Model


class CNNNet:
    @staticmethod
    def buildNet():
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(128, 128, 3)))
        # model.add(layers.MaxPool2D(2,2))

        model.add(layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))

        model.add(layers.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))

        model.add(layers.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dense(units=28, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildFineTuneNet():
        model = models.Sequential()
        conv_base = VGG16(input_shape=(128, 128, 3),
                          weights='imagenet',
                          include_top=False)

        for layer in conv_base.layers:
            print(layer.name)
            if 'block5' in layer.name:
                layer.trainable = False

        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(units=28, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildResNet():

        regs = l1_l2()
        model_in = layers.Input(shape=(128, 128, 3))

        # Basic model build...
        model_body = layers.SeparableConv2D(16, (3, 3), kernel_regularizer=regs, padding='same')(model_in)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(16, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.SeparableConv2D(32, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(32, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.SeparableConv2D(64, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(64, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.SeparableConv2D(128, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(128, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.SeparableConv2D(256, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(256, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization(axis=1)(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(256, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization(axis=1)(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.SeparableConv2D(512, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(512, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.SeparableConv2D(512, (3, 3), kernel_regularizer=regs, padding='same')(model_body)
        model_body = layers.BatchNormalization()(model_body)
        model_body = layers.Activation('relu')(model_body)
        model_body = layers.MaxPool2D((2, 2))(model_body)

        model_body = layers.Flatten()(model_body)

        model_out = layers.Dropout(0.5)(model_body)
        model_out = layers.Dense(256, activation='relu', kernel_regularizer=regs)(model_out)
        model_out = layers.Dropout(0.5)(model_out)
        model_out = layers.Dense(28, activation='sigmoid', kernel_regularizer=regs)(model_out)

        model = Model(model_in, model_out)
        return model

