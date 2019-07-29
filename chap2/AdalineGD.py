#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 07:17:13 2019

@author: amit
"""

import numpy as np
class AdalineGD(object):
    """ ADaptive LINear NEuron classifier.
    
    Parameters
    ----------
    eta : float
         Learning rate (between 0.0 and 1.0)
    n_iter : int
         Passes over the training dataset
    
    Attributes
    ----------
    w_ : 1d-array
        weights after fitting
    errors_ : list
        Number of misclassificcations in every epoch
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
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
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 # sum of squared error for ADALINE
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        ''' calculate the net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        ''' Compute Linear activation '''
        return self.net_input(X)
        
    def predict(self, X):
        ''' Return class label after unit step '''
        return np.where(self.activation(X) >= 0.0, 1, -1)
          
        