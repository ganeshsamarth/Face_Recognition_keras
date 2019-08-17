from resnet import *
import keras
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from metrics import ArcFace
from keras.utils import to_categorical
import h5py

import data_manip

x_train = data_manip.X
y_train = to_categorical(data_manip.Y, num_classes=100)

inputs = keras.layers.Input(shape=(64, 64, 3))
label = keras.layers.Input(shape=(100,))

x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(512, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
output = ArcFace(n_classes=100)([x, label])

model = Model([inputs, label], output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit([x_train, y_train],
          y_train,
          batch_size=4,
          epochs=1,
          verbose=1,
          callbacks=[ModelCheckpoint('model.hdf5',
                     verbose=1, save_best_only=True)])