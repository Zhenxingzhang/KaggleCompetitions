# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:04:14 2014

@author: sku
"""
from pre_process import *
from log_regression import *   
import numpy as np
import pandas as pd

def submit(filename):
    """(string) -> None
    
    Runs logistic regression on the titanic dataset and generates
    a submission file 'filename'.csv
    
    """
    #set up logistic regression parameters
    alpha=0.1
    n_iter=1000
    lambd=0
    threshold=0.5
    #read data
    train,test=read_data()
    #does some of the trevor stephens pre-processing, check docstring of function
    train,test,y_matrix,id_matrix=pre_process_(train,test)

    train.info()

    print(train["title=Dr"].value_counts())
    print(train["title=Master"].value_counts())
    print(train["title=Miss"].value_counts())
    print(train["title=Mr"].value_counts())
    print(train["title=Mrs"].value_counts())
    print(train["title=Rev"].value_counts())
    print(train["boss"].value_counts())
    print(train["lady"].value_counts())
    #turns dataframes into matrices so the logistic regression code doesnt flip out
    train_,test_=turn_into_matrices(train,test)
    #get theta
    _,theta=logistic_regression(train_,y_matrix,alpha,n_iter,lambd)
    print theta
    #get predictions in the test set
    pred=predict(test_,theta)
    #merge PassengerId and predictions to create the submission DataFrame
    df=pd.DataFrame(np.concatenate([id_matrix,pred],axis=1))
    #set up column names
    df.columns=['PassengerId','Survived']
    #turn PassengerId column from float to int
    df['PassengerId']=df['PassengerId'].astype('int')
    #turn predictions from float into binary int 0,1
    df['Survived']=(df['Survived']>threshold)*1
    #create submission csv
    df.to_csv(filename,header=True,index=False)

submit("Zhenxing_Titanic_Result.csv")