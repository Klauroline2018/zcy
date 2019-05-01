# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:18:59 2019

@author: ZCY
"""

import pandas as pd
df_train = pd.read_csv('file:///C:/Users/ZCY/Desktop/Lecture Notes/Introduction to Machine Learning/task2_s92hdj/train.csv')
df_test = pd.read_csv('file:///C:/Users/ZCY/Desktop/Lecture Notes/Introduction to Machine Learning/task2_s92hdj/test.csv')
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


def categorisation_accuracy(a,b):
    return accuracy_score(a,b)

Xprimal = np.array(df_train.iloc[:,2:22])
y = np.array(df_train.iloc[:,1])
Xtestprimal = np.array(df_test.iloc[:,1:21])

kf = KFold(n_splits=10)

normalizer = preprocessing.Normalizer().fit(Xprimal)
Xnormalization = normalizer.transform(Xprimal)
Xtestnormalization = normalizer.transform(Xtestprimal)

selector=SelectKBest(mutual_info_classif, k=12)
ss = StandardScaler()
Xselection = selector.fit_transform(Xnormalization, y)
X = ss.fit_transform(Xselection)
Xtestselection = selector.transform(Xtestnormalization)
X_test=ss.transform(Xtestselection)

acc = 0
for train_index, test_index in kf.split(X):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    ytrain = y[train_index]
    ytest = y[test_index]
    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(20,), shuffle = True, random_state = None)
    clf.fit(Xtrain,ytrain) 
    ypredit = clf.predict(Xtest)
    acc = acc + categorisation_accuracy(ytest,ypredit)
    average_acc = acc/10
print(average_acc)

y_predit = clf.predict(X_test)
row=y_predit.shape[0]
seq=np.array(range(2000,2000+row))
data = pd.DataFrame({"Id":seq,"y":y_predit})
data.to_csv('./res.csv',index=False,header=True)
print(y_predit)




