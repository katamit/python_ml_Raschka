#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 06:43:28 2019

@author: amit
"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02,test_idx=None):
#     setup marker generator and color map
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
#     plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1 , X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1 , X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contour(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx1.min(), xx2.max())
    
#     plot class labels
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=cmap(idx),
                   marker=markers[idx], label=cl,
                   edgecolor='black')
    
#    highlight test samples
    if test_idx:
        #Plot all samplesx
        X_test, y_test = X[test_idx,:] , y[test_idx]
        
        plt.scatter(X_test[:,0], X_test[:,1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test_set')
        
        
        
        
        
        
def get_iris(standarized=True):
    '''
    Provides the iris data set with train and test split
    X of the dataset contain two featues only : petal length and 
    petal width
    
    set standarized =False , to get non -standarized data set
    '''
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    Y = iris.target
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,  random_state=1, stratify=Y)
    
    if standarized:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        
    return X_train, X_test, Y_train, Y_test
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        