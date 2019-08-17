from resnet2 import *
import keras
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from metrics import ArcFace
from keras.utils import to_categorical
import h5py
import numpy as np

import data_manip

# h5f1 = h5py.File("/home/svp/AI/facebank2/training_data.h5", 'r')
# train_X = h5f1['X']
# train_Y = h5f1['Y']
# h5f1.close()

train_X = data_manip.X
train_X = np.array(train_X)
train_Y = np.array(data_manip.Y)

train_Y = to_categorical(train_Y, num_classes=100)
model = ResNet18((64,64,3),100)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(train_X,
          train_Y,
          batch_size=4,
          epochs=1,
          callbacks=[ModelCheckpoint('model.hdf5',
                     verbose=1, save_best_only=False)])