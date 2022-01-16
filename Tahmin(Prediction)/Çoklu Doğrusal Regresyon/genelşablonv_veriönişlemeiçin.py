# Kütüphaneleri ekle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#veriyi yükle

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)


boy = veriler [["boy"]]
print(boy)

boykilo = veriler[["boy","kilo"]]

print(boykilo)


## eksik verilerin işlenmesi için impute kullanımı

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
# :,1:4
Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])

Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)


ulke = veriler.iloc[:,0:1].values
print(ulke)

# Encoder nominalden ordinal(kategorik verilerden) e bir numeric dönüş sağlıyor 
from sklearn import preprocessing
# sayısal encoder

le = preprocessing.LabelEncoder()


ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

#kolon basliklara etiklere tasimak ve 1 0 yerlestirmek
ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

#numpy dizileri dataframe e dönüstürme

sonuc = pd.DataFrame(data = ulke , index = range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data =Yas,index=range(22),columns=["boy","kilo","yas"])

cinsiyet = veriler.iloc[:,-1].values

print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ["cinsiyet"])
#data frame birlestirme islemi
s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)


s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)

#verileri eğitim ve test icin bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)
# verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler



X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)













































































