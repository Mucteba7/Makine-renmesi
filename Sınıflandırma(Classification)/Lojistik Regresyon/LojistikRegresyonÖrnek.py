import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


veriler = pd.read_csv("veriler.csv")
veriler.drop([0,1,2,3,4],axis="index",inplace = True) # Burada doğruluğu artırmak adına küçük yaşlı, çocukların indexleri çıkarıldı!!
print(veriler)



x=veriler.iloc[:,1:4].values #bağımsız değişken

y = veriler.iloc[:,4:] #bağımlı değişken
print(veriler.corr())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) 

from sklearn.linear_model import  LogisticRegression 

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train)



y_pred = logr.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)
