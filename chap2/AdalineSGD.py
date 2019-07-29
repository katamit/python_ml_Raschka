#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 06:24:46 2019

@author: amit
"""

import numpy as np
class AdalineSGD(object):
    """ ADaptive LINear NEuron classifier.
    
    Parameters
    ----------
    eta : float
         Learning rate (between 0.0 and 1.0)
    n_iter : int
         Passes over the training dataset
    shuffle: boolean (default : True)
         shuffles data every epoch if True
         to PREVENT CYCYLES.
    random_state : int
         Radom number generator seed for random weights
         initialization.
    
    
    Attributes
    ----------
    w_ : 1d-array
        weights after fitting
    cost_ : list
        Sum-of-squares cost function value averaged over all training examples
        in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit training data
        Parameters
        ----------
        
        X : {array-like}, shape = [n_samples, n_features]
        Trainig vectores, where n_samples is the number of samples
        and n_features is the number of variables
        
        y: {array-like}. shape = [n_samplees]
        Target values
         
        Returns
        --------
        
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X,y):
        '''fit training data without reinitializing the weights'''
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        '''Shuffle traing data'''
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        '''Initialize weights to small random number'''
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        '''Apply adaptive learning rule to update the weights'''
        output = self.activation(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        ''' calculate the net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        ''' Compute Linear activation '''
        return self.net_input(X)
        
    def predict(self, X):
        ''' Return class label after unit step '''
        return np.where(self.activation(X) >= 0.0,1, -1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
          
        