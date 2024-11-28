# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:26:47 2024

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas  as  pd

#import data set
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding categorical data 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],     remainder='passthrough')                         # Leave the rest of the columns untouched

 
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
X = np.array(ct.fit_transform(X), dtype= np.float64)



#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 3]= labelencoder_X.fit_transform(X[:, 3 ])
#onehotencoder = OneHotEncoder(categorical_features=[3])
#X = onehotencoder.fit_transform(X).toarray()

#evitar la trampa de las variables ficticias 
#X = X[:, 1:]

#splining the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)      

#ajustar el modelo de regresion lineal multiple con el congunto de entrenamiento 
from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(X_train, y_train)

#prediccion de los resultados en el congunto de testing 
y_pred = regression.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))

#construir un modelo de eliminacion hacia atras para importrla 
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
SL= 0.05 

X_opt = X[:, [0, 2, 3, 4, 5, 6] ]
print(X_opt.dtype)
X_opt = sm.add_constant(X_opt)
regression_OLS = sm.OLS(endog = y,  exog = X_opt).fit()
regression_OLS.summary()






 