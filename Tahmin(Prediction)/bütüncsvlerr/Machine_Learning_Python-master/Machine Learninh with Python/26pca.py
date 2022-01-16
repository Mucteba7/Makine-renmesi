# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:19:58 2019

@author: Cumali
"""


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



veriler = pd.read_csv('Wine.csv')


X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values



x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

pca=PCA(n_components=2)

X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

y_pred=classifier.predict(X_test)

y_pred2=classifier2.predict(X_test2)

cm=confusion_matrix(y_test,y_pred)

cm2=confusion_matrix(y_test,y_pred2)

cm3=confusion_matrix(y_pred,y_pred2)

print(cm)
print("******************")
print(cm2)
print("------------------------------")
print(cm3)

lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

y_pred_lda=classifier_lda.predict(X_test_lda)

cm4=confusion_matrix(y_pred,y_pred_lda)

print(cm4)




 

