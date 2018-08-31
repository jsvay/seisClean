import os
import random
import keras as k
import numpy as np


class DataGen(k.utils.Sequence):
    def __init__(self, directory, img_rows=1126, img_cols=636,
        batch_size=8, n_channels=1, n_classes=2, shuffle=True, resize=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.directory = directory
        self.numfiles = os.listdir(self.directory)
        self.list_directory = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize = resize
        self.on_epoch_end()


    def __len__(self):
        "Sets the number of batches per epoch"
        return int(np.floor(len(self.numfiles) / self.batch_size))


    def __getitem__(self, index):
        "Generates data"
        X = np.empty((self.batch_size, self.img_rows, self.img_cols, self.n_channels))
        print(X.shape)
        for i in range(self.batch_size):
            f = self.list_directory.pop()
            tmp_data = np.load(self.directory +f).T
            temp = tmp_data[::self.resize,::self.resize]
            X[i,] = np.expand_dims(tmp_data, -1)
        return X, X


    def on_epoch_end(self):
        "Refreshing data list on epoch end"
        self.list_directory = self.numfiles
        if self.shuffle == True:
            np.random.shuffle(self.list_directory)
