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

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color = "red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color = "blue")

