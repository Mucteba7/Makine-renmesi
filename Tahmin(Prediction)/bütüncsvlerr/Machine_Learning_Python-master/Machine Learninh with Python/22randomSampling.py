# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:33:35 2019

@author: Cumali
"""

import pandas as pd
import random 
import matplotlib.pyplot as plt

data=pd.read_csv("Ads_CTR_Optimisation.csv")

N=10000
d=10
summary=0
choosens=[]
for n in range(0,N):
    ad=random.randrange(d)
    choosens.append(ad)
    reward=data.values[n,ad]#verilerdeki n. satır 1 ise ödül 1
    summary=summary+reward

plt.hist(choosens)
plt.show()    