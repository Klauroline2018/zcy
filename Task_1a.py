# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:52:51 2019

@author: ZCY
"""

import pandas as pd
df_train = pd.read_csv('file:///C:/Users/ZCY/Desktop/Lecture Notes/Introduction to Machine Learning/train.csv')
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import KFold
import math

def loss_function(a,b):
    number=len(a)
    return math.sqrt(np.sum((a-b)**2)/number)

alpha_set=[0.1,1,10,100,1000]
X = df_train.iloc[:,2:12].values
y = df_train.iloc[:,1].values
kf = KFold(n_splits=10)

for i in alpha_set:
    RMSE = 0
    for train_index, test_index in kf.split(X):
        Xtrain = X[train_index,:]
        Xtest = X[test_index,:]
        ytrain = y[train_index]
        ytest = y[test_index] 
        reg = linear_model.Ridge(alpha=i)
        reg.fit(Xtrain,ytrain)
        reg.coef_
        ypredict=reg.predict(Xtest)
        RMSE = RMSE + loss_function(ytest,ypredict)
        average_RMSE=RMSE/10
    print(average_RMSE)
    

  
   

    
