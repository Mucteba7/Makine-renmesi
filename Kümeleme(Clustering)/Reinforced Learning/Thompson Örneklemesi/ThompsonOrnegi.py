import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


veri = pd.read_csv("Ads_CTR_Optimisation.csv")

N= 10000 #10bin tıklama
d = 10 #10 ilan

toplam = 0 #toplam odul
secilenler = []
birler = [0]*d #formülden gelen 10luk bir liste
sıfırlar = [0]*d #formülden gelen 10luk liste
for n in range(0,N): # Her bir ilan tıklamasını simüle ediyoruz.
    ad = 0 # seçilen ilan
    max_th = 0
    for i in range(0,d):# Her bir ilanın teker teker ihtimalleri her ilan tıklamasını bulmaya yarayan döngü yani en yüksek ucbye sahip ilanı bulmak
        rasbeta = random.betavariate(birler[i]+1, sıfırlar[i]+1)    
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veri.values[n,ad]
    if odul == 1:
        birler[ad]=birler[ad] +1
    else:
        sıfırlar[ad]=sıfırlar[ad]+1
    
    toplam = toplam + odul    
    
print("Toplam Ödül : ",toplam)
plt.hist(secilenler)
plt.show()
