import numpy as np
import os
import shutil
from PIL import Image

read_dir='D:\\IMFDB_final'
output_dir='C:\\Users\\ganeshsamarth\\Desktop\\facebank2'
os.mkdir(output_dir)
actor_list=os.listdir(read_dir)
for actors in actor_list:
    if actors=='.DS_Store':
        continue
    else:
        dir=read_dir+'\\'+actors
        os.mkdir(output_dir+'\\'+actors)
        j=1
        actor_movies=os.listdir(dir)
        for movies in actor_movies:
            if movies=='.DS_Store':
                continue
            else :
                movie_dir=dir+'\\'+movies
                image_dir=movie_dir+'\\'+'images'
                for images in os.listdir(image_dir):
                    shutil.copy(image_dir+'\\'+images,output_dir+'\\'+actors)
                    os.rename(output_dir+'\\'+actors+'\\'+images, output_dir+'\\'+actors+'\\'+actors+'_'+str(j)+'.jpg')
                    img=Image.open(output_dir+'\\'+actors+'\\'+actors+'_'+str(j)+'.jpg')
                    img=img.resize((64,64))
                    img.save( output_dir+'\\'+actors+'\\'+actors+'_'+str(j)+'.jpg')
                    j+=1
