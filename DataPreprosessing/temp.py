# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import library like padas nu 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pdc:%5CUsers%5Cdavit%5Cworkspace%5CMaching_Learning%5CMachine Learning-A-Z-Codes-Datasets%5CMachine Learning A-Z (Codes and Datasets)%5CPart 1 - Data Preprocessing%5CSection 2 -------------------- Part 1 - Data Preprocessing --------------------%5CPython%5Cdata_preprocessing_template.pyc:%5CUsers%5Cdavit%5Cworkspace%5CMaching_Learning%5CMachine Learning-A-Z-Codes-Datasets%5CMachine Learning A-Z (Codes and Datasets)%5CPart 1 - Data Preprocessing%5CSection 2 -------------------- Part 1 - Data Preprocessing --------------------%5CPython%5Cdata_preprocessing_tools.ipynbc:%5CUsers%5Cdavit%5Cworkspace%5CMaching_Learning%5CMachine Learning-A-Z-Codes-Datasets%5CMachine Learning A-Z (Codes and Datasets)%5CPart 1 - Data Preprocessing%5CSection 2 -------------------- Part 1 - Data Preprocessing --------------------%5CPython%5Cdata_preprocessing_tools.py

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#tratamiento de los N/A
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #el 0 para hacer lo por columna 1 para fila
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
 
#Encoding categorical data
#Encoding the independient variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder ='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding the independient variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)






 





 






















