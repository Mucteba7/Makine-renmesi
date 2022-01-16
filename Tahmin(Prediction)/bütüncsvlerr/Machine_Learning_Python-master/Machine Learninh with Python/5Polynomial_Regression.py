# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:08:58 2019

@author: Cumali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#veri yükleme
veri=pd.read_csv("maaslar.csv")
#data frame  dlimleme (slice)
x=veri.iloc[:,1:2]
y=veri.iloc[:,2:]

#numpy arraye çevirme
X=x.values
Y=y.values


#linear Regression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()
#Polynomal Regression
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()


poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

#tahminler


print(lin_reg.predict([[10]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
