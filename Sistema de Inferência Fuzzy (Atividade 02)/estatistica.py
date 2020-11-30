# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:39:36 2020

@author: Arthur Pires
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import numpy as np


data = load_iris() 
data0 = np.array(data.data)

# Normalização das dimensões pela média
for i in range (len(data0[0,:])):
    data0[:,i]*=(1/np.mean(data0[:,i]))

# Separando os dados pelas classes
# data1=data0[0:50]
# data2=data0[50:100]
# data3=data0[100:]

# Dividindo dataset iris em 2
data1=data0[0:25]
data2=data0[50:75]
data3=data0[100:125]


# Plot dos dados
bplot1=plt.boxplot(data1,patch_artist=True)
bplot2=plt.boxplot(data2,patch_artist=True)
bplot3=plt.boxplot(data3,patch_artist=True)  

c="pink"
colors = [c,c,c,c]
    
for patch, color in zip(bplot2['boxes'], colors):
     patch.set_facecolor(color)

c="lightgreen"
colors = [c,c,c,c]

for patch, color in zip(bplot3['boxes'], colors):
     patch.set_facecolor(color)


    
plt.show()




# %config InlineBackend.figure_format = 'retina'


maxl=max(data.data[:,2])
maxw=max(data.data[:,3])
minl=min(data.data[:,2])
minw=min(data.data[:,3])
