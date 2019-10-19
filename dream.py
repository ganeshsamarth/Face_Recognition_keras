#DREAM custom implementation
import cv2
import numpy as np
from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda,multiply, add,PReLU
import tensorflow as tf


# load embeddings
x=np.load('x_array_50.npy')
label=np.load('y_array_50.npy')
print(x.shape)
#defining final yaw calculation
def sep_x_yaw(x):
    sep_x = np.empty((0,128))
    sep_yaw = np.empty((0,128))
    for i in range(len(x)):
        print(i)
        temp = np.reshape(x[i],(1,128))
        if (i % 2 == 0):
            
            sep_x = np.append(sep_x,temp, axis = 0)
        else:
            sep_yaw = np.append(sep_yaw,temp, axis = 0)
    return [sep_x,sep_yaw]

def yaw_coeff(yaw):
    #print(yaw.shape)
    value=np.multiply(0.022,abs(yaw))
    sig_value=1./(1.+np.exp(-value))
    return sig_value

data_sep = sep_x_yaw(x)
data_sep_x = data_sep[0]
data_sep_yaw = data_sep[1] 
data_sigma = yaw_coeff(data_sep_yaw)
print(data_sigma.shape)
# neural network definition
data_x=Input(shape=(128,))
data_yaw = Input(shape=(128,))

#print(layer2.shape)
output = Dense(128)(data_x)
output = PReLU()(output)
outputs = Dense(128)(output)
outputs = PReLU()(outputs)
outputs = Dropout(0.4)(outputs)

y = data_yaw
layer2 = multiply([outputs,y])
final_embedding = add([data_x,layer2])

model=Model(inputs=[data_x,data_yaw],outputs=final_embedding)

# training

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
print(data_sep_yaw.shape)
model.fit([data_sep_x,data_sigma],label,batch_size=128,epochs=150)
model.save('dream_model.h5')



