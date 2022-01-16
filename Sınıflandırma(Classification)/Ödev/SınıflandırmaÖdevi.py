import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import f1_score as f1
veriler = pd.read_csv("Iris.csv")



veriler["Species"].unique
label_encoder = preprocessing.LabelEncoder()
veriler['Species'] = label_encoder.fit_transform(veriler['Species'])


veriler['Species'].unique()


x=veriler.iloc[:,1:5].values #bağımsız değişken



y = veriler.iloc[:,5:] #bağımlı değişken


print(veriler.corr())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0) # y_test = y_true

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) 
############################# Logistic Regresyon
from sklearn.linear_model import  LogisticRegression 

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train)

y_pred_logistic = logr.predict(X_test)
cm_logistic = confusion_matrix(y_test,y_pred_logistic)
print("Logistic Regresyon: ")
print(cm_logistic)
print("R2 Skor",r2_score(y_test,y_pred_logistic))
print("F1 Skor : ",f1(y_test,y_pred_logistic,average="macro"))


######################################## Karar Ağacı
from sklearn.tree import DecisionTreeClassifier 

dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train,y_train)
y_pred_decision = dtc.predict(X_test)
cm_decision = confusion_matrix(y_test,y_pred_decision)
print("Decision Tree: ")
print(cm_decision)
print("R2 Skor",r2_score(y_test,y_pred_decision))
print("F1 Skor : ",f1(y_test,y_pred_decision,average="macro"))
################################# SVC(SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel="rbf") ### linear olarak değiştirebilirsin!
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test,y_pred_svc)
print("SVM Classifier: ")
print(cm_svc)
print("R2 Skor",r2_score(y_test,y_pred_svc))
print("F1 Skor : ",f1(y_test,y_pred_svc,average="macro"))
##################### Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
cm_gnb = confusion_matrix(y_test,y_pred_gnb)
print("Naive Bayes Gaussian: ")
print(cm_gnb)
print("R2 Skor",r2_score(y_test,y_pred_gnb))
print("F1 Skor : ",f1(y_test,y_pred_gnb,average="macro"))
################################ Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
y_pred_nn = neigh.predict(X_test)
cm_nn = confusion_matrix(y_test,y_pred_nn)
print("En yakın Komşu algoritması: ")
print(cm_nn)
print("R2 Skor",r2_score(y_test,y_pred_nn))
print("F1 Skor : ",f1(y_test,y_pred_nn,average="macro"))



