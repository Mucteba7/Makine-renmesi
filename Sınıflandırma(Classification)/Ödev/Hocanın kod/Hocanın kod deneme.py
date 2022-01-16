import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score as f1

from sklearn.metrics import confusion_matrix

veriler = pd.read_excel("Iris.xls") 

x = veriler.iloc[:,0:4]

y = veriler.iloc[:,4:]
print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0)
log.fit(X_train,y_train)
y_pred_logistic = log.predict(X_test)

cm = confusion_matrix(y_test,y_pred_logistic)
print(cm)
print("R2 Skor",r2_score(y_test,y_pred_logistic))
print("F1 Skor : ",f1(y_test,y_pred_logistic,average="macro"))
from sklearn.svm import SVC
svc = SVC(kernel = "poly")
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred_svc)
print(cm1)
