# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:23:04 2022

@author: Michael Augustine
"""

import pandas as pd 
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv('IRIS.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x ='species', data = iris)
plt.show()
sns.scatterplot('sepal_length','sepal_width', hue= 'species', data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1),hue= 'species', height=2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x = iris.corr(method= 'pearson')
print(x)