# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:52:51 2019

@author: ZCY
"""

import pandas as pd
df_train = pd.read_hdf("train.h5","train")
df_test = pd.read_hdf("test.h5","test")
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def categorisation_accuracy(a,b):
    return accuracy_score(a,b)

def normalization(x):
    output = np.copy(x)
    m=[]
    s=[]
    for i in range(x.shape[1]):
        mean_val = np.mean(x[:,i])
        std_val=np.std(x[:,i])
        output[:,i]=(x[:,i]-mean_val)/std_val
        m.append(mean_val)
        s.append(std_val)
    return output

Xorigin = np.array(df_train.iloc[:,1:121])
y = np.array(df_train.iloc[:,0])
Xtestorigin = np.array(df_test.iloc[:,0:120])

kf = KFold(n_splits=10)

Xnormalize = normalization(Xorigin)
Xtestnormalize = normalization(Xtestorigin)

selector=SelectKBest(mutual_info_classif, k=80)
ss = StandardScaler()
Xselect = selector.fit_transform(Xnormalize, y)
X = ss.fit_transform(Xselect)
Xtestselect = selector.transform(Xtestnormalize)
X_test=ss.transform(Xtestselect)

acc = 0
n_splits=10
for train_index, test_index in kf.split(X):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    ytrain = y[train_index]
    ytest = y[test_index]
    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(100,100,100,100,100), shuffle = True, random_state = None)
    clf.fit(Xtrain,ytrain) 
    ypredit = clf.predict(Xtest)
    acc = acc + categorisation_accuracy(ytest,ypredit)
average_acc = acc/n_splits
print(average_acc)

y_predit = clf.predict(X_test)
row=y_predit.shape[0]
seq=np.array(range(45324,45324+row))
data = pd.DataFrame({"Id":seq,"y":y_predit})
data.to_csv('./res.csv',index=False,header=True)
print(y_predit)
