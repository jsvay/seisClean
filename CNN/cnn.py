import os
import keras as k
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gen import DataGen


class basic_CNN():
    def __init__(self, labels, imgs_rows, img_cols, channels=1, nclasses=2,
                batch_size=16):
        self.labels = labels
        self.img_rows = imgs_rows
        self.img_cols = img_cols
        self.channels = channels
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.nfilters = 16
        self.model = k.models.Sequential()


    def conv_layers(self, nfilters, f_size=3, stride=2, pad='valid',
                    input_shape=None):
        """Creating layer object for the network. This requiers less code than
        having to redo"""
        if input_shape != None:
            self.model.add(k.layers.convolutional.Conv2D(nfilters,
                            kernel_size=f_size, strides=stride,
                            padding=pad, input_shape=input_shape))
        else:
            self.model.add(k.layers.convolutional.Conv2D(nfilters,
                            kernel_size=f_size, strides=stride, padding=pad))
        self.model.add(k.layers.Activation("tanh"))
        return self


    def execution(self):
        self.conv_layers(self.nfilters, input_shape=self.img_shape)
        self.model.add(k.layers.MaxPooling2D(pool_size=2, strides=2))
        self.conv_layers(2*self.nfilters)
        self.model.add(k.layers.MaxPooling2D(pool_size=2, strides=2))
        self.conv_layers(4*self.nfilters)
        self.model.add(k.layers.MaxPooling2D(pool_size=2, strides=2))
        self.conv_layers(8*self.nfilters)
        self.model.add(k.layers.Flatten())
        self.model.add(k.layers.Dense(500, activation='relu'))
        self.model.add(k.layers.Dense(self.nclasses, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='Adam',
                            metrics=['accuracy'])
        self.model.summary()
        return


    def train(self, lenData, lenValid, epochs, train_dir, valid_dir, factor, shuffle):
        params = {'batch_size': self.batch_size,
                  'img_cols': self.img_cols,
                  'img_rows': self.img_rows,
                  'n_classes': self.nclasses,
                  'n_channels': self.channels,
                  'shuffle': shuffle}

        training_generator = DataGen(self.labels, train_dir, **params, factor=factor)
        validation_generator = DataGen(self.labels, valid_dir, **params)
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


def run():
    #General params
    labels = {'n':0, 'y':1}
    img_rows = 1126
    img_cols = 636

    #Train params
    lenData = 1600
    lenValid = 300
    batch_size = 64
    epochs = 4
    train_dir = '/s0/SI/train/'
    valid_dir = '/s0/SI/valid/'
    factor = (0.1, 3.0)
    shuffle = True

    #Creating class object, which is the network
    CNN = basic_CNN(labels, img_rows, img_cols, batch_size=batch_size)
    CNN.execution()
    CNN.train(lenData, lenValid, epochs, train_dir, valid_dir, factor, shuffle)

    k.backend.clear_session()


if __name__ == '__main__':
    run()
