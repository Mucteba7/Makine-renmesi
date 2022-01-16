import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

'''
x = veriler.iloc[:,[2,4]].values # 2. ve 4. kolonları almak istersek. !! Önemli!!
'''
x = veriler.iloc[:,3:].values # burada tüm satırları al, 3. kolon ve sonrasını al yani hacim ve maaş kolonlarını al

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters =3,init ="k-means++")
kmeans.fit(x)
clusters =kmeans.cluster_centers_
print(clusters)


#burada en optimum k değerini hesaplamak için grafik çizdirerek grafik üzerinden elbow noktasını bulduk k=2 veya k=3 veya k=4
#alınabilir.
sonuclar= []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init ="k-means++",random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)


plt.plot(range(1,11),sonuclar)