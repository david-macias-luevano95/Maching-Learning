# Maching Learning 
## Data Preprosesing proses
```

# import library like padas nu 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
```

## Lineal regresion
```
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # clase para cargar los datos 

# Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

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
