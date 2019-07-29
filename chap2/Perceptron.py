# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 07:17:13 2019

@author: amit.
"""

import numpy as np
class Perceptron(object):
    """ Perceptron classifier.
    
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
        Trainig vectroes, where n_samples is the number of samples
        and n_features is the number of variables
        
        y: {array-like}. shape = [n_samplees]
        Target values
         
        Retrurns
        --------
        
        self : object
        """
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi , target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        ''' calculate the net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        ''' Return class label after unit step '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
          
        