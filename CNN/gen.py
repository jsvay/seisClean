import os
import random
import keras as k
import numpy as np


class DataGen(k.utils.Sequence):
    def __init__(self, labels, directory, img_rows=1126, img_cols=636,
        batch_size=8, n_channels=1, n_classes=2, shuffle=True, factor=(1,1)):
        self.labels = labels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.directory = directory
        self.numfiles = os.listdir(self.directory)
        self.list_directory = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.factor=factor
        self.on_epoch_end()


    def __len__(self):
        "Sets the number of batches per epoch"
        return int(np.floor(len(self.numfiles) / self.batch_size))


    def __getitem__(self, index):
        "Generates data"
        X = np.empty((self.batch_size, self.img_rows, self.img_cols, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        for i in range(self.batch_size):
            tmp = np.round(random.uniform(self.factor[0], self.factor[1]), 2)
            f = self.list_directory.pop()
            tmp_data = np.load(self.directory + f).T
            X[i,] = np.expand_dims(tmp_data*tmp, -1)
            y[i] = self.labels[f[0]]
        return X, k.utils.to_categorical(y, num_classes=self.n_classes)


    def on_epoch_end(self):
        "Refreshing data list on epoch end"
        self.list_directory = self.numfiles
        if self.shuffle == True:
            np.random.shuffle(self.list_directory)
