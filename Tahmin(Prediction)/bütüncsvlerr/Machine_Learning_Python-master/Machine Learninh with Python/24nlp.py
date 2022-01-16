# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:24:47 2019

@author: Cumali
"""

import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


data=pd.read_csv("Restaurant_Reviews.csv")

nltk.download("stopwords")
ps=PorterStemmer()

derlem=[]
for i in range(1000):
    # "^" not anlamı getiriyor yani a dan z ye olmayanları bul ve boşlukla değiştir 
    yorum=re.sub("[^a-zA-Z]"," ",data["Review"][i])
    
    #kelime karmaşıklığını önlemek için hepsini küçülttük 
    yorum=yorum.lower()
    yorum=yorum.split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    
    yorum=" ".join(yorum) 
    derlem.append(yorum)
#   BAG OF WORDS
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(derlem).toarray() #bağımsız değişken    
y=data.iloc[:,1].values #bağımlı değişken

#Machine Learning
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)
    