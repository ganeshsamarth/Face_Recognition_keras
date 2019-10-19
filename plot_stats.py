import matplotlib.pyplot as plt
import numpy as np

with open('/home/ganesh/Desktop/class_acc.txt','r') as myfile:
    data = myfile.read()
    actor_list=list()
    accuracy_list=list()
    x = data.strip().split('\n')
    for i in range(len(x)):
        y=x[i].split(',')
        print(y)
        actor_list.append(y[0])
        accuracy_list.append(y[1])
    myfile.close()

plt.scatter(np.arange(0,len(accuracy_list)),accuracy_list)
