from resnet import *
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from metrics import ArcFace
input=Input(shape=(64,64,3))
label=Input(shape=(100,))
batch_size=8
epochs=10
x=ResNet18((64,64,3),1000)
x = Dense(512, kernel_initializer='he_normal')(x)
x=BatchNormalization(x)
output=ArcFace(num_classes=100)[x,label]
model = Model([input, label], output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit([x_train, y_train],
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([x_test, y_test], y_test),
          callbacks=[ModelCheckpoint('model.hdf5',
                     verbose=1, save_best_only=True)])
