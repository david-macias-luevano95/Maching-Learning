# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:04:07 2024

@author: davit
"""

#svr
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
y = y.reshape(len(y),1)



#scale of features 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)  #cuando se quere crear el modelo 
y = sc_Y.fit_transform(y)  #cuando se quiere mostrar el modelo 
y = y.flatten()



#ajustar la regresion con el dataset 
from sklearn.svm import SVR
regression = SVR(kernel='rbf')
regression.fit(X,y)
 
#prediccion de nuestros modelos con SVR
y_pred = regression.predict([6.5])    #se transforma el 6.5 para saver si se entienden los parametros entre fit transform fit solo se calcula la modificacion 

#fit cuando se tiene el modelo transforcuando transformarlo 
#tener cuidato de ajudtar los argumentos de la funcion normalizado diferentes 
#terminos dentro de las variables 
#solo e fit se transforma pero no se aplica 
#   

#visualizacion de los resultados del SVR 
#Xgrdi = np.arange(min(X) )
#X_grid=  X_grid .reshape(min(X), max(X), 0.1)
plt.scatter(X, y, color= 'red')
plt.plot(X, regression.predict(X), color= 'blue')
plt.xlabel('position of employe')
plt.ylabel('Sueldo en $')
plt.show()





