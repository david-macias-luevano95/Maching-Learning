# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:44:33 2025

@author: davit
"""
#Regresion con Arboles de desicion

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1]


from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, y)



y_pred = regression.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict(X), color = 'blue')
plt.title('Regresion Tree Decission')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() #dentro del