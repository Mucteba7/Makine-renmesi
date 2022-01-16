# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:20:36 2019

@author: Cumali
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix



veriler = pd.read_csv('Churn_Modelling.csv')


X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values


le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]


x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



classifier=Sequential()
classifier.add(Dense(6,init="uniform",activation="relu",input_dim=11))
classifier.add(Dense(6,init="uniform",activation="relu"))
classifier.add(Dense(1,init="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])
classifier.fit(X_train,y_train,epochs=50)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)


cm=confusion_matrix(y_test,y_pred)
print(cm)








