import os
import shutil
import numpy as np
import random

train = '/s0/SI/train/'
test = '/s0/SI/test/'
valid = '/s0/SI/valid/'

"""
ranger = np.linspace(0, 999, 1000)
ranger = ranger.astype(int)
ranger = ranger.tolist()
random.shuffle(ranger)


for x in range(150):
    item = ranger.pop()
    shutil.move(train + str(item) + '.npy', valid)

for x in range(50):
    item = ranger.pop()
    shutil.move(train + str(item) + '.npy', test)
"""

files = os.listdir(train)

y = []
n = []

for file in files:
    if file.startswith('n'):
        n.append(file)
    else:
        y.append(file)


random.shuffle(n)
random.shuffle(y)

a = [n, y]

for i in a:
    for x in range(150):
        item = i.pop()
        shutil.move(train + item, valid)

    for x in range(50):
        item = i.pop()
        shutil.move(train + item, test)
