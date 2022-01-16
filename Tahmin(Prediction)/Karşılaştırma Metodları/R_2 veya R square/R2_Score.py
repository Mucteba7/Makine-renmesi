from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,-1:]
X = x.values
Y = y.values
############################################# Linear regresyon
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x.values,y.values)
plt.scatter(x.values,y.values,color = "red")
plt.plot(x,lin_reg.predict(x),color  = "blue")
plt.show()

############################################# polinomal regresyon 2.dereceden

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X,Y,color = "red")
a = plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show(a)
############################################################### 4.dereceden

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X,Y,color = "red")
a = plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show(a)


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)

plt.scatter(X,Y,color = "red")
plt.plot(X,r_dt.predict(X),color = "blue")
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
                    