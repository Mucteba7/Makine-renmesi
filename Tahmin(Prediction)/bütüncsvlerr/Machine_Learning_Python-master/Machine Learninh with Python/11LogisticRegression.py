# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:08:02 2019

@author: Cumali
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data=pd.read_csv("veriler.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

logr=LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)
print(y_pred)
