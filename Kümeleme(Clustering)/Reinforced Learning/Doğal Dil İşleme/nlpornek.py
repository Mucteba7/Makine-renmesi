


import numpy as np
import pandas as pd

veri = pd.read_csv("Restaurant_Reviews.csv")

import re
# Veri temizleme Preprocessing
yorum = re.sub('[^a-zA-Z]'," ",veri["Review"][0]) #alfanümerik karakterlerin filtrelenmesi.

yorum = yorum.lower() # bütün harfleri küçülttük
yorum = yorum.split() #listelemek için harfleri

import nltk #stopwordsleri listemizden atmak için gerekli olan kütüphaneyi import ettik



from nltk.stem.porter import PorterStemmer #kelimelerin çekim eklerini atmak için,gövdelerine indirgemek için

ps = PorterStemmer()


# Veri temizleme Preprocessing
from nltk.corpus import stopwords
derleme =[]
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]'," ",veri["Review"][i])
    yorum = yorum.lower() 
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum) # yorumu al boşluklarla birleştir bir string oluştur
    derleme.append(yorum)
    
#Kelime vektörü Sayaç Kullanımı (CountVectorizer),Öznitelik çıkarımı
#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000) #en fazla kullanılan 1000 kelimeyi al daha fazlasını alma

x = cv.fit_transform(derleme).toarray() #bağımsız değişken hem train ediyor hem dönüştürüyor ve diziye çeviriyor
y = veri.iloc[:,-1:]   #bağımlı değişken

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)#yüzde seksen train yüzde 20 test için kullanılcak veri

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train,y_train) # yani x train verisinden y train verisini öğren aralarındaki bağlantıyı bul
y_pred = gnb.predict(x_test) #yüzde 20 lik testten tahminlerini çıkar bakalım

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) # gerçek veri ile tahmin verisini karşılaştır
print("Accuracy Oranı : (%)" ,100 *(cm[0][0]+cm[1][1]) /(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) )
