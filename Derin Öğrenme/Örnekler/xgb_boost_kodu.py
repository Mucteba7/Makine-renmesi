from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Ön İşleme
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import  ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] =labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] =labelencoder_X_1.fit_transform(X[:, 2])

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')

X = ct.fit_transform(X)

X= ct.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)


