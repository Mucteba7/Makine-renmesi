import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


veriler = pd.read_csv("veriler.csv")




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

y_pred1 = logr.predict(X_test)


from sklearn.metrics import confusion_matrix
'''
cm = confusion_matrix(y_test,y_pred)
print(cm)
'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski") # n yi 1 girersek detaya ineriz ve toplam 8 veriden 7 sini doğru tahmin eder.

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
############################ Kernel trick için önce svm i ekliyoruz 
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
cm1 = confusion_matrix(y_test,y_pred_svc)
print("SVC : ", cm1)
