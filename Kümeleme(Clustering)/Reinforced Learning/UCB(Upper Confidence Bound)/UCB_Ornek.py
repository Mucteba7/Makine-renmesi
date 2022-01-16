import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


veri = pd.read_csv("Ads_CTR_Optimisation.csv")

N= 10000 #10bin tıklama
d = 10 #10 ilan
#Ri(n)
oduller = [0]*d #her bir elemanı 0 olan 10 elemanlı liste,yani ilk başta bütün ilanların ödülü 0 
#Ni(n)
tiklamalar = [0]*d
toplam = 0 #toplam odul
secilenler = []

for n in range(0,N): # Her bir ilan tıklamasını simüle ediyoruz.
    ad = 0 # seçilen ilan
    max_ucb = 0
    for i in range(0,d):# Her bir ilanın teker teker ihtimalleri her ilan tıklamasını bulmaya yarayan döngü yani en yüksek ucbye sahip ilanı bulmak
        if(tiklamalar[i]>0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb = ortalama + delta        
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veri.values[n,ad]
    oduller[ad] = oduller[ad]+odul
    toplam = toplam + odul
    
print("Toplam Ödül : ",toplam)
plt.hist(secilenler)
plt.show()