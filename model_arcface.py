from resnet import *
import keras
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from metrics import ArcFace
import h5py


h5f1 = h5py.File("/home/svp/AI/facebank2/training_data.h5", 'r')
train_X = h5f1['X']
train_Y = h5f1['Y']
h5f1.close()

inputs=keras.layers.Input(shape=(64,64,3))
label= keras.layers.Input(shape=(100,))

batch_size=8
epochs=10
x = ResNet18((64,64,3),1000)
x = Dense(512, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
output=ArcFace(n_classes=100)([x,label])
model = Model([inputs, label], output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit([train_X, train_Y],
          train_Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[ModelCheckpoint('model.hdf5', verbose=1, save_best_only=False)]
        )

# .{epoch:02d}