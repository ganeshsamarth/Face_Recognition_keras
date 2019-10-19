#DREAM custom implementation
import cv2
import numpy as np
from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda,Multiply, Add
import tensorflow as tf


# load embeddings
x=np.load('x_array.npy')
y=np.load('y_array.npy')

#defining final yaw calculation
def sep_x_yaw(x):
    sep_x = np.empty((0,128))
    sep_yaw = np.empty((0,128))
    for i in range(0,len(x)):
        if (i % 2 == 0):
            sep_x = np.append(sep_x,x[i], axis = 0)
        else:
            sep_yaw = sep_yaw.append(sep_yaw,x[i], axis = 0)
    return [sep_x,sep_yaw]

def yaw_coeff(yaw):
    value=(((4/180)*yaw) -1 )
    sig_value=1./(1.+tf.math.exp(-value))
    return sig_value

data_sep = sep_x_yaw(x)
data_sep_x = data_sep[0]
data_sep_yaw = data_sep[1]    

# neural network definition
data_x=Input(shape=(128,))
data_yaw = Input(shape=(128,))

#print(layer2.shape)
output = Dense(128,activation='relu')(data_x)
outputs = Dense(128, activation='relu')(output)
outputs = Dropout(0.4)(outputs)

y = data_yaw
layer2 = Multiply([outputs,y])
final_embedding = Add([data_x,layer2])

model=Model(inputs=[output.input,y.input],outputs=final_embedding)

# training

model.compile(optimizer='rmsprop',loss='mean_squared_error')

model.fit([data_sep_x,data_sep_yaw],y,batch_size=2,epochs=30)
model.save('dream_model.h5')



