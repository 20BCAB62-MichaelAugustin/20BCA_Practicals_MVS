# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:54:03 2022

@author: Michael Augustine
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



data=pd.read_csv('Telecom_Data.csv')

       
data.info()


# regressor variables 
x = data.iloc[:, 0:20].values
#print(x)
  
# regressed variables
y = data.iloc[:, 20].values
#print(y)


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

