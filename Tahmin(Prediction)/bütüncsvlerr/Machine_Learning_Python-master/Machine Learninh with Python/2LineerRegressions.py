# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 17:41:40 2019

@author: Cumali
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

veriler=pd.read_csv("satislar.csv")
print(veriler)

aylar=veriler[["Aylar"]]
satışlar=veriler[["Satislar"]]

x_train,x_test,y_train,y_test=train_test_split(aylar,satışlar,test_size=0.33,random_state=0)
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("aylara göre satış tahmini")
plt.xlabel("aylar")
plt.ylabel("satışlar")