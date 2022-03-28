# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:06:09 2022

@author: Michael Augustine
"""
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 


dataset = pd.read_csv('CarPrice_Assignment.csv')
dataset.drop(['car_ID','CarName'],axis=1,inplace=True)
dataset.info()
df = dataset.iloc[:, [8,9]].values

Z = linkage(df, method = "ward")
dendro = dendrogram(Z)
plt.title('Dendogram')
plt.ylabel('Euclidean distance')
plt.show()
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

labels = ac.fit_predict(df)
plt.figure(figsize = (8,5))
plt.scatter(df[labels == 0,0] , df[labels == 0,1], c= 'red')
plt.scatter(df[labels == 1,0] , df[labels == 1,1], c= 'blue')
plt.scatter(df[labels == 2,0] , df[labels == 2,1], c= 'green')
plt.scatter(df[labels == 3,0] , df[labels == 3,1], c= 'black')
plt.scatter(df[labels == 4,0] , df[labels == 4,1], c= 'orange')
plt.show()
