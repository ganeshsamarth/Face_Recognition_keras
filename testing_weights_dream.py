#Testing weights file for dream_test
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
import pickle


image_dir_basepath = '/home/ganesh/Downloads/IMFDB_Align'
names = os.listdir(image_dir_basepath)
image_size = 160
dream_path='/home/ganesh/Face_Rec_triplet/notebook/dream_model.h5'
model_path = '/home/ganesh/keras-facenet/model/keras/facenet_keras.h5'
model = load_model(model_path)
dream_model = load_model(dream_path)
yaw_dict = {}
with open('/home/ganesh/Desktop/yaw_values.txt','r') as myfile:
    data=myfile.read()
    yaw_data_list=list()
    actor_list=list()
    frontal_list=list()
    profile_list=list()
    x=data.strip().split('\n')
    for i in range(len(x)):
        y=x[i]
        z=y.split(',')
        actor=z[0].split('_')[0]
        yaw=float(z[1])
        yaw_dict[image_dir_basepath + '/' + actor + '/' + z[0]] = yaw
    myfile.close()

def prewhiten(x,yaw_list):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y,yaw_list

def yaw_coeff(yaw):
    #print(yaw.shape)
    value=np.multiply(0.022,abs(yaw))
    sig_value=1./(1.+np.exp(-value))
    return sig_value

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin):
    #cascade = cv2.CascadeClassifier(cascade_path)
    yaw_list= list()
    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img,(160,160))
        yaw_list.append(yaw_dict[filepath])
        aligned_images.append(np.array(img))
            
    return np.array(aligned_images),yaw_list

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images,yaw_list = load_and_align_images(filepaths, margin)
    aligned_images,yaw_list = prewhiten(aligned_images,yaw_list)
    pd = []
    final_embedding=np.empty((0,128))
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    final_embedding = np.empty((0,128))
    for i in range(embs.shape[0]):
        pre_embed = embs[i,:]
        pre_embed = np.reshape(pre_embed,(1,128))
        yaw_array = np.full((1,128),yaw_coeff(yaw_list[i]))
        embed_final = dream_model.predict([pre_embed,yaw_array])
        final_embedding = np.append(final_embedding,embed_final,axis = 0)
    print(final_embedding.shape)
    return final_embedding
def infer(le, clf, filepaths):
    embs = calc_embs(filepaths)
    pred = le.inverse_transform(clf.predict(embs))
    return pred

le = pickle.load(open('LabelEncoder.sav','rb'))
clf = pickle.load(open('SVC.sav','rb'))
acc=list()

for names in os.listdir(image_dir_basepath):
    test_file_paths = [os.path.join(image_dir_basepath+'/'+names,images) for images in os.listdir(image_dir_basepath+'/'+names)]
    pred = infer(le, clf, test_file_paths)
    count=0
    #print(pred)
    for items in pred:
        if items==names:
            count+=1
    print(names+':'+ str(count/len(pred)))
    with open('/home/ganesh/Desktop/class_acc.txt','a+') as myfile3:
        myfile3.write(names+','+str(acc)+'\n')
        myfile3.close()
    acc.append(count/len(pred))
print(sum(acc)/len(acc))
