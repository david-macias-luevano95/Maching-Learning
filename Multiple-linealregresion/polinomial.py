# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:30:04 2024

@author: davit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the data set 
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values 

#ajustar la regresion lineal con el dataset 
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#ajustar la regresion polinomica con el dataset
#transformacion de la nueva matris de caracteristicas 
#se eleva al cuadrado los valores que tiene la variable independiente 
#anadiendo la colunma de 1 se utiliza por muchas librerias 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures( degree= 2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#visualizacion de los resultados del midelo lineal



#visualizacion de regresion polinimica con el data set
plt.scatte( X, y, color= 'read')









