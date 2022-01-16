# Kütüphaneleri ekle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#veriyi yükle

veriler = pd.read_csv('satislar.csv')

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]



#verileri eğitim ve test icin bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0)

'''
# verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler



X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)
Y_train = StandardScaler().fit_transform(y_train)
Y_test = StandardScaler().fit_transform(y_test)
'''
#model inşası(linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)  


tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))