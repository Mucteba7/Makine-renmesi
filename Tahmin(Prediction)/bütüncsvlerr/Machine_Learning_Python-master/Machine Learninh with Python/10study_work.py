# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:20:36 2019

@author: Cumali
"""

import pandas as pd 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


data=pd.read_csv("maaslar_yeni.csv")

x=data.iloc[:,2:5]
y=data.iloc[:,5:]
X=x.values
Y=y.values

print(data.corr())
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print("POLY-------------------------")
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("DECİSİONNN TREEE************************")
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)
model3=sm.OLS(dt_reg.predict(X),X)
print(model3.fit().summary())


print("RANDOM FOREST-------")
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
model4=sm.OLS(rf_reg.predict(X),X)
print(model4.fit().summary())

