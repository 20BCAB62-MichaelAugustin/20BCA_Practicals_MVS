# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:09:14 2022

@author: Michael Augustine
"""

import pandas as pd
from factor_analyzer import FactorAnalyzer
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt


df=pd.read_csv('CarPrice_Assignment.csv')
df.info()
df.drop(['car_ID','CarName'],axis=1,inplace=True)
df.info()
# Converting the categorical data into continous was done manually using FIND AND REPLACE in MS Excel.

# Checking the correlation
x= df.corr(method= 'pearson')
print(x)
sns.heatmap(df.corr(method='pearson'),data=df)
plt.show()
# Bartlett’s test
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)


# Kaiser-Meyer-Olkin (KMO) Test
kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)
# KMO values range between 0 and 1. Value of KMO less than 0.5 is considered inadequate.
# The overall KMO for our data is 0.78, which is pretty good. 
# This value indicates that we can proceed with our planned factor analysis.


#Choosing the number of factors
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.analyze(df, 25, rotation=None)
#Check Eigenvalues
ev, v = fa.get_eigenvalues()
print(ev)

fa = FactorAnalyzer()
fa.fit(df)
eigen_values, vectors = fa.get_eigenvalues()
print(vectors)
# 3 eigen values are greater than 1 therefore,
# NUMBER OF FACTORS = 3


# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),vectors)
plt.plot(range(1,df.shape[1]+1),vectors)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
# It is understandable from the scree plot that the number of factors 3 or 4.

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=6, rotation='varimax')
fa.fit(df)
loadings = fa.loadings_
print(loadings)


# Get variance of each factors
print(fa.get_factor_variance())
# It is in the below format 
#                      Factor 1       Factor2       Factor3 
# SS Loadings
# Proportion Var
# Cummulative Var

# Total 58% cumulative Variance is explained by the 3 factors.
