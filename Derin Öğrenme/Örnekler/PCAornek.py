import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler  = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Wine.csv")#githubdan raw olarak aldık.

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)
#PCA Dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
#Pca dönüşümünden sonra gelen lr
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#pca olmadan çıkan cm 
cm = confusion_matrix(y_test,y_pred)
print("Gerçek ve PCA'siz\n", cm)
 
print("===========================================")
 
cm2 = confusion_matrix(y_test,y_pred2)
print("Gerçek ve PCA'li :\n", cm2)

print("==========================================")

cm3= confusion_matrix(y_pred,y_pred2)
print("Pcasiz ve Pcali:\n",cm3)