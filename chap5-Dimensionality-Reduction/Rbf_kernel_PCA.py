#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:37:51 2019

@author: amit
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF Kernel PCA implementation.
    
    
    Parameter
    ---------
    X: (Numpy ndarray), shape = [n_samples, n_features]
    
    gamma : float
      Tuning parameter of the RBF Kernel
      
      n_components: int
        Number of principal components to return
        
    Returns:
    -----------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
      projected dtaset
      
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional datset.
    sq_dists = pdist(X, 'sqeuclidean')
    
    #Convert pariwise distances into a square matrix.
    
    mat_sq_dists = squareform(sq_dists)
    
    #Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    
    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N))  / N
    K = K - one_n.dot(K)  - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    
    #Obtaining eigenpairs form the centered kernel matrix
    # scipy.linalg.eigh return them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    
    #Collect the top k eighenvectos   (projected samples)
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    
    return X_pc    
    