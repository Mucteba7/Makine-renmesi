import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler = pd.read_csv('maaslar_yeni.csv')
x = veriler.iloc[:,2:5] # bu 3 kolon alıyor bizim aslında tek kolon almamız lazım o da ; iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values # X gerçek değeri
Y = y.values # Y gerçek değeri

print(veriler.corr()) # Burada kolerasyon matrisini kullanarak hangi kolonların birbiriyle daha net ilişkide olduğunu anlayıp ona göre kolon seçimi yapmak gerekir.
#yani tek parametre almak(tek kolon almak yani o da Ünvan seviyesi kolonu) r square değerini artıracaktır.

# Lineer Regresyon Modeli eğitimi ve skor karşılaştırması
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
model = sm.OLS(lin_reg.predict(X),X) ##bu OLS ler çok önemli oradaki r square değeri dikkate alınmalı!
print(model.fit().summary())
print("Linear R2 degeri: ",r2_score(Y, lin_reg.predict((X))))


#Polinomal Regresyon eğitimi ve skor karşılaştırması
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print("Polinomal R^2 Değeri : ",r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

# Destek Vektör Regresyonu ve Ölçekleme (Support Vector Regression , Scaling)
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print("SVR Model R^2 Değeri : ",r2_score(y_olcekli, svr_reg.predict(x_olcekli)))
# Karar Ağacı ile Tahmin (Decision Tree)
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,np.ravel(Y))
print('dt ols')
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
print("Decision Tree R2 degeri: ",r2_score(Y, r_dt.predict(X)))
# Rassal Orman (Random Forest)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,np.ravel(Y))
print('dt ols')
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
print("Random Forest R2 degeri: ",r2_score(Y, rf_reg.predict(X)))
print("=============================================================")
print("Linear R2 degeri: ",r2_score(Y, lin_reg.predict((X))))
print("Polinomal R^2 Değeri : ",r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))
print("SVR Model R^2 Değeri : ",r2_score(y_olcekli, svr_reg.predict(x_olcekli)))
print("Decision Tree R2 degeri: ",r2_score(Y, r_dt.predict(X)))
print("Random Forest R2 degeri: ",r2_score(Y, rf_reg.predict(X)))
print("=============================================================")

