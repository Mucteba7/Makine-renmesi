import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#sci-kit learn 
from sklearn.preprocessing import Imputer


missings=pd.read_csv('eksikveriler.csv')
#print(missings)

imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
Yas=missings.iloc[:,1:4].values
#print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)
