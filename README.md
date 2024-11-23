# Maching Learning 
## Data Preprosesing proses
lineal regresion asismet for determine 

## Polinomial Regression

```
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#training the linea regresion model inthe whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#training the lineal regresion for polinomial regresion 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualization the lineal regresion results 
plt.scatter(X, y, color= 'red')
plt.plot(X, lin_reg.predict(X), color= 'blue')
plt.title('lineal Regresion Model')
plt.xlabel('posision del empleado')
plt.ylabel('Salary in $')
plt.show()

#visualization the polynomial regesion model 
X_grid= np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X,  lin_reg_2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.title('Modelo de regresion polynomial ')
plt.xlabel('posision del empleado')
plt.ylabel('Salaty in $')
plt.show()

```
