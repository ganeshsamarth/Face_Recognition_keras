import os
from PIL import Image
import random
import numpy as np
import h5py
input_dir='/home/svp/AI/facebank2'

dataset=list()

for i,actors in enumerate(os.listdir(input_dir)):
    items_list=list()
    for images in os.listdir(input_dir+'/'+actors):
        items_list.append(input_dir+'/'+actors+'/'+images)
        items_list.append(i)
        print(i)
        dataset.append(items_list)

print(len(dataset))
random.shuffle(dataset)

X=[]
Y=[]
count=0
for line in dataset:
    path=line[0]
    label=line[1]
    img=Image.open(path)
    np_image=np.array(img)
    X.append(np_image)
    Y.append(label)
    count+=1
    print(count)



# h5f = h5py.File(input_dir+'/'+'training_data.h5', 'w')
# h5f.create_dataset("X", data=X)
# h5f.create_dataset("Y", data=Y)
# h5f.close()
