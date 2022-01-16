
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


veri = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000
d = 10
toplam  = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veri.values[n,ad] # verilerdeki n. satır = 1 ise odul 1 artar
    toplam = toplam + odul


# çizim
plt.hist(secilenler)
plt.show()