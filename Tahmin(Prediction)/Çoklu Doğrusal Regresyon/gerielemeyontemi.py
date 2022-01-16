
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing






veriler = pd.read_csv('odev_tenis.csv')
print(veriler)
windy = veriler.iloc[:,3:4].values
print(windy)

temp = veriler.iloc[:,1:2].values
humidity = veriler.iloc[:,2:3].values
print(humidity)
print(temp)


#encoder: Kategorik -> Numeric
outlook = veriler.iloc[:,0:1].values
print(outlook)


le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
windy[:,0] = le.fit_transform(veriler.iloc[:,0])


print(windy)
print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
windy = ohe.fit_transform(windy).toarray()
print(outlook)
print(windy) ### windyi dummy variable cinsinden al

play = veriler.iloc[:,-1:].values

le = preprocessing.LabelEncoder()

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(play)





#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['overcast','rainy','sunny'])
print(sonuc)

sonuc3 = pd.DataFrame(data=play, index = range(14), columns = ['play'])
print(sonuc3)

sonuc2 = pd.DataFrame(data = windy[:,:1], index = range(14), columns = ['windy'])
print(sonuc2)
sonuc4 = pd.DataFrame(data=humidity,index = range(14),columns = ['humidity'])
sonuc5 = pd.DataFrame (data = temp , index  = range(14), columns =['tempature'])

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc4], axis=1)
print(s)
s1 = pd.concat([s,sonuc5],axis = 1)
s2=pd.concat([s1,sonuc2], axis=1)
s3 = pd.concat([s2,sonuc3],axis = 1)
print(s3)
'''
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)


#BACKWARD ELLIMINATION

import statsmodels.api as sm

X = np.append( arr = np.ones((22,1)).astype(int), values=veri,axis=1)

X_l  = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype = float)

model = sm.OLS(boy,X_l).fit()

print(model.summary())
     '''                                                           
                                                                  
