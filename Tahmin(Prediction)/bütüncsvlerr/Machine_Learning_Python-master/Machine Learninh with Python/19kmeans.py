# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:57:38 2019

@author: Cumali
"""

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("musteriler.csv")

X=data.iloc[:,3:].values

kmeans=KMeans(n_clusters=3 ,init="k-means++")
kmeans.fit(X)
sonuclar=[]
print(kmeans.cluster_centers_)

for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,10),sonuclar)