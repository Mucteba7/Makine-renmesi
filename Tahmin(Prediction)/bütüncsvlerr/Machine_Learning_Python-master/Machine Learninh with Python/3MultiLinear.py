# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:18:48 2019

@author: Cumali
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


veriler = pd.read_csv('veriler.csv')





Yas = veriler.iloc[:,1:4].values


ulke = veriler.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)

le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

#cinsiyet = veriler.iloc[:,-1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_prediction=regressor.predict(x_test)


boy=s2.iloc[:,3:4].values
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]

veri=pd.concat([sol,sag],axis=1)



x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_prediction2=r2.predict(x_test)

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_list=veri.iloc[:,[0,1,2,3,4,5,]].values

r=sm.OLS(endog=boy,exog=X_list).fit()

print(r.summary())

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_list=veri.iloc[:,[0,1,2,3,5,]].values

r=sm.OLS(endog=boy,exog=X_list).fit()

print(r.summary())





















