# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:55:05 2024

@author: David
"""


import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:. 1:2].values
y = dataset.iloc[:, 2].values

#adjust the lineal regresion for this model 
