# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:59:14 2019

@author: Cumali
"""

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

data=pd.read_csv("musteriler.csv")

X=data.iloc[:,3:].values

ac=AgglomerativeClustering(n_clusters=3 ,affinity="euclidean",linkage="ward")
y_pred=ac.fit_predict(X)

print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,color="red")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,color="blue")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,color="green")
plt.show()

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()