import numpy as np
import keras as k
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
import os
import tensorflow as tf
from tqdm import tqdm
from ae_gen import DataGen


class basic_AE():
    def __init__(self, imgs_rows, img_cols, channels=1, nclasses=2,
                batch_size=16):
        self.img_rows = imgs_rows
        self.img_cols = img_cols
        self.channels = channels
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.nfilters = 64
        input_img, decoded, self.model = self.structure()


    def conv_layers(self, input, nfilters=False, sfilter=(3,3),
                        activation='relu', padding='same'):
        if not nfilters:
            nfilters = self.nfilters
        x = Conv2D(nfilters, sfilter, activation=activation, padding=padding)(input)
        return x



    def structure(self):
        input_img = Input(shape=(self.img_rows, self.img_cols, self.channels))
        x = self.conv_layers(input_img, activation='tanh')
        x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        x = self.conv_layers(x)
        x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        x = self.conv_layers(x)
        encoded = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        ### Latent space
        x = self.conv_layers(encoded)
        x = UpSampling2D((2, 2))(x)
        x = self.conv_layers(x)
        x = UpSampling2D((2, 2))(x)
        x = self.conv_layers(x)
        x = UpSampling2D((2, 2))(x)
        decoded = self.conv_layers(x, nfilters=1, activation='sigmoid')
        decoder = Model(input_img, decoded)
        return input_img, decoded, decoder


    def compiler(self):
        self.model.compile(optimizer='Adam', loss='binary_crossentropy',
                                metrics=['accuracy'])
        self.model.summary()


    def train(self, lenData, lenValid, epochs, train_dir, valid_dir, factor,
                                                            shuffle, resize):
        params = {'batch_size': self.batch_size,
                  'img_cols': self.img_cols,
                  'img_rows': self.img_rows,
                  'n_classes': self.nclasses,
                  'n_channels': self.channels,
                  'shuffle': shuffle,
                  'resize': resize}

        training_generator = DataGen(train_dir, **params)
        validation_generator = DataGen(valid_dir, **params)
        tensorboard = k.callbacks.TensorBoard(log_dir='./trainlogs', histogram_freq=0,
                                write_graph=True, write_images=True)

        self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch=int(lenData/self.batch_size),
                    validation_steps=int(lenValid/self.batch_size),
                    epochs=epochs,
                    workers=6,
                    callbacks=[tensorboard])

        decoded_imgs = self.model.predict(training_generator)
        rows = int(self.img_rows)#/resize)
        cols = int(self.img_cols)#/resize)
        n = 4
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i)
            plt.imshow(training_generator[i].reshape(rows, cols))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i].reshape(rows, cols))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('output.png')


def run():
    #General params
    labels = {'n':0, 'y':1}
    resize = 12
    img_rows = int(1126/4.39)
    img_cols = int(636/4.95)

    #Train params
    lenData = 1600
    lenValid = 300
    batch_size = 64
    epochs = 10
    train_dir = '/s0/SI/train/'
    valid_dir = '/s0/SI/valid/'
    factor = (0.1, 3.0)
    shuffle = True

    #Creating class object, which is the network
    AE = basic_AE(img_rows, img_cols, batch_size=batch_size)
    AE.compiler()
    #AE.train(lenData, lenValid, epochs, train_dir, valid_dir, factor, shuffle, resize)

    k.backend.clear_session()


if __name__ == '__main__':
    run()
