import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


## You can use the following library for clustering
from sklearn.cluster import KMeans

## read from csv file
data = pd.read_csv('C:\\.....Your path to csv data file')

##The following code displays the shape of data set
print(data.shape)
print(data.head())

##Extracting each column
f1= data['Sports']
f2 = data['Religious']
f3 = data['Nature']
f4 = data['Theatre']
f5 = data['Shopping']
f6 = data['Picnic']

## Creating an array of data points
X = np.array(list(zip(f1,f2,f3,f4,f5,f6)))

## Write your code to perform K-Means clustering algorithm with different parameters
## Your program should also print clustering labels
## You can also refer to SKleran python package to know more about programming API's for generating labels and clustering visualization

''' Your code goes here'''

##Write code to visualize the clusters
'''Your code goes here'''


'''You can also use your own implementation instead of the sample code shown above'''
