# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 15:44:44 2014

@author: sku
"""

import numpy as np
import pandas as pd

def hyp_log_r(X,theta):
    """(matrix,vector) -> vector
    
    Takes dataset 'X' and parameter vector 'theta' and
    returns logistic regression hypothesis/model.
    
    >>> theta=np.array([[1],[2],[3]])
    >>> extend_target_variable(y)
    >>> X=np.array([[ 1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0.,  1.]])
    >>> theta=np.array([[1.],[-1.],[0.]])
    >>> hyp_log_r(X,theta)
    array([[ 0.73105858],
           [ 0.26894142],
           [ 0.5       ]])

    """
    return 1./(1.+np.exp(-(np.dot(X,theta))))
    
def cost(h,y,theta,lambd):
    """(vector,vector,vector,number) -> number
    
    Takes hypothesis 'h', target variable vector 'y', 
    parameter vector 'theta' and regularization parameter number 'lambd'
    and returns the logistic regression cost function.
    
    >>> h=np.array([[.5],[1.],[0.]])
    >>> y=np.array([[1.],[1.],[0.]])
    >>> theta=np.array([[1.],[1.],[0.]])
    >>> lambd=0
    >>> cost(h,y,theta,lambd)
    0.23774928408898272

    """
    m=np.shape(y)[0]
    #first term of cost function
    j_1=float(np.dot(y.T,np.log(h)))
    #second term of cost function
    j_2=float(np.dot(1-y.T,np.log(1-h)))
    #check andrew ng's regularization lecture
    reg=((lambd/(2*m))*np.sum(np.power(theta[1:],2)))
    return -((1./m)*(j_1+j_2))+reg
    
def normalize_features(X):
    """(matrix) -> matrix
    
    Takes dataset 'X' and returns a dataset with features normalized
    so that they have mean 0 and standard deviation 1. 
    
    >>> X=np.array([[ 1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0.,  1.]])
    >>> normalize_features(X)
    array([[ 1.41421356, -0.70710678, -0.70710678],
           [-0.70710678,  1.41421356, -0.70710678],
           [-0.70710678, -0.70710678,  1.41421356]])

    """
    #number of columns of X
    n=np.shape(X)[1]
    #subtract from each column its mean and divide by its standard deviation
    for j in range(n):
        X[:,j]=(X[:,j]-X[:,j].mean())/(X[:,j].std())
        
    return X

def grad_descent(theta,h,y,X,alpha,n_iter,lambd):
    """(vector,vector,vector,number,number,number) -> vector,vector
    
    Takes parameter vector 'theta', hypothesis vector 'h', 
    target variable vector 'y',  dataset 'X', 
    learning rate parameter number 'alpha', number of iterations 'n_iter',
     and regularization parameter number 'lambd'
    to compute the gradient descent applied to the logistic regression model.
    Returns hypothesis vector 'h' with the update parameters 'theta'
    and the updated parameter vector 'theta' itself.
    
    *** Example on the logistic_regression function ***

    """
    #initialize list of lists 'thetas' and 'costs' (ended up not using costs)
    thetas=[]
    costs=[]
    #number of rows of y
    m=np.shape(y)[0]
    #run n_iter times
    for i in range(n_iter):
        #update first value(bias) of theta
        theta[0]=theta[0]-((alpha/m)*np.sum(h-y))
        #update rest of paramters of theta (regularized version)
        theta[1:]=theta[1:]-((alpha/m)*np.dot(X[:,1:].T,(h-y))-((lambd/m)*theta[1:]))
        #computes hypothesis with updated parameter theta (to remove)        
        h=hyp_log_r(X,theta)
        #computes hypothesis with updated parameter theta 
        #note: to use with learning curves
        j=cost(h,y,theta,lambd)
        #append updated 'theta' and 'cost' in list of lists 'thetas' and 
        #'costs'
        thetas.append(theta)
        costs.append(j)
        
    return h,theta
    
def logistic_regression(X,y,alpha,n_iter,lambd):
    """(matrix,vector) -> vector
    
    Takes dataset 'X' and parameter vector 'theta'
    to logistic regression hypothesis/model. 
    Returns prediction vector 'pred'
    
    >>> threshold=0.5
    >>> lambd=0
    >>> n_iter=10000
    >>> alpha=0.1
    >>> d=pd.read_table("./data/ex2data1.txt",sep=",",header=None)
    >>> X=d[d.columns[0:2]].as_matrix();
    >>> y=np.matrix(d[d.columns[2]]).T
    >>> h,theta=logistic_regression(X,y,alpha,n_iter,lambd)
    >>> print theta
    >>> print 'Training Accuracy: '+str(float(sum((h>=threshold)==y)))+'%'
    Training Accuracy: 89.0%


    """
    #number of rows of X
    m=np.shape(X)[0]
    #normalize features of X
    X=normalize_features(X)
    #concatenate column of ones (bias)
    X=np.concatenate((np.ones((m,1)),X),axis=1)
    #number of columns of X
    n=np.shape(X)[1]
    #initialize theta with ones
    theta=np.ones((n,1))
    #compute initial hypothesis
    h=hyp_log_r(X,theta)
    #run gradient descent to obtain both the updated hypothesis and parameter
    #vector
    h,theta=grad_descent(theta,h,y,X,alpha,n_iter,lambd)
    
    return h,theta
    
def predict(X,theta):
    """(matrix,vector) -> vector
    
    Takes dataset 'X' and parameter vector 'theta'
    to logistic regression hypothesis/model. 
    Returns prediction vector 'pred'.
    
    *** Example on the logistic_regression function ***

    """
    #number of rows of X
    m=np.shape(X)[0]
    #normalize features of X
    X=normalize_features(X)
    #concatenate column of ones (bias)
    X=np.concatenate((np.ones((m,1)),X),axis=1)
    #compute hypothesis
    pred=hyp_log_r(X,theta)
    
    return pred


        
