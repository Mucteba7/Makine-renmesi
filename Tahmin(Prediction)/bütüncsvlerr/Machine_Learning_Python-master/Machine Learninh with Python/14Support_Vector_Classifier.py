# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:20:56 2019

@author: Cumali
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data=pd.read_csv("veriler.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

svc=SVC(kernel='rbf')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print(cm)


