# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:02:10 2019

@author: Cumali
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sp

veri=pd.read_csv("odev_tenis.csv")

'''
play=veri.iloc[:,-1:].values
le=LabelEncoder()
play[:,0]=le.fit_transform(play[:,0])

windy=veri.iloc[:,3:4].values
windy=le.fit_transform(windy)
'''

veriler2=veri.apply(LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]
ohe=OneHotEncoder()
c=ohe.fit_transform(c).toarray()

hava_durumu=pd.DataFrame(data=c, index=range(14),columns=["Overcast","Sunny","Rainy"])
last_data=pd.concat([hava_durumu,veri.iloc[:,1:3]],axis=1)
last_data=pd.concat([last_data,veriler2.iloc[:,-2:]],axis=1)

humidity=last_data.iloc[:,4:5]
without_hum=last_data.iloc[:,[0,1,2,3,5,6]]

x_train,x_test,y_train,y_test=train_test_split(without_hum,humidity,test_size=0.33,random_state=0)

regression=LinearRegression()
regression.fit(x_train,y_train)

y_predict=regression.predict(x_test)

#X=np.append(arr=np.ones((14,1)).astype(int),values=without_hum,axis=1)

X_list=without_hum.iloc[:,[0,1,2,3,4,5]].values

r_ols=sp.OLS(endog=without_hum,exog=X_list)

r=r_ols.fit()

print(r.summary())