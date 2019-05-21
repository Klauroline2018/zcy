# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:49:12 2019

@author: ZCY
"""

import pandas as pd
df_train_labeled = pd.read_hdf("train_labeled.h5","train")
df_train_unlabeled = pd.read_hdf("train_unlabeled.h5","train")
df_test = pd.read_hdf("test.h5","test")
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LeakyReLU

train_labeled = np.array(df_train_labeled.iloc[:,1:140])
y_labeled = np.array(df_train_labeled.iloc[:,0])
train_unlabeled = np.array(df_train_unlabeled.iloc[:,0:139])
test = np.array(df_test.iloc[:,0:139])

normalizer = preprocessing.Normalizer().fit(train_labeled)
train_labeled_norm = normalizer.transform(train_labeled)
normalizer = preprocessing.Normalizer().fit(train_unlabeled)
train_unlabeled_norm = normalizer.transform(train_unlabeled)

kf = KFold(n_splits=10)

y_onehot = keras.utils.to_categorical(y_labeled, 10)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_labeled_norm, y_onehot, epochs=200, batch_size=128)
y_predit = model.predict(train_unlabeled_norm)
y_predit1 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_labeled_norm, y_onehot, epochs=200, batch_size=128)
y_predit = model.predict(train_unlabeled_norm)
y_predit2 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_labeled_norm, y_onehot, epochs=200, batch_size=128)
y_predit = model.predict(train_unlabeled_norm)
y_predit3 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_labeled_norm, y_onehot, epochs=200, batch_size=128)
y_predit = model.predict(train_unlabeled_norm)
y_predit4 = np.argmax(y_predit,axis=1)

y_predict = []
for i in range(len(y_predit1)):
    y_predict.append(np.bincount([y_predit1[i],y_predit2[i],y_predit3[i],y_predit4[i]]).argmax())
y_unlabeled = np.array(y_predict)

df_train_unlabeled = df_train_unlabeled.insert(0, 'y', y_unlabeled)

df_train = df_train_labeled.append(df_train_unlabeled)

train = np.array(df_train.iloc[:,1:140])
y = np.array(df_train_labeled.iloc[:,0])

normalizer = preprocessing.Normalizer().fit(train)
train_norm = normalizer.transform(train)
normalizer = preprocessing.Normalizer().fit(test)
test_norm = normalizer.transform(test)

selector=SelectKBest(mutual_info_classif, k=120)
ss = StandardScaler()
Xselection = selector.fit_transform(train_norm, y)
X = ss.fit_transform(Xselection)
Xtestselection = selector.transform(test_norm)
X_test=ss.transform(Xtestselection)

def categorisation_accuracy(a,b):
    return accuracy_score(a,b)

acc = 0
n_splits=10
count = 0
for train_index, test_index in kf.split(X):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    ytrain = y_onehot[train_index,:]
    ytest = y_onehot[test_index,:]
    model = Sequential()
    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Xtrain, ytrain, epochs=1, batch_size=64)
    ypredit = model.predict(Xtest)
    ypredit_1 = np.argmax(ypredit,axis=1)
    ytest_1 = np.argmax(ypredit,axis=1)
    acc = acc + categorisation_accuracy(ytest_1,ypredit_1)
    count += 1
    if count == 1:
        break
    
average_acc = acc/n_splits
print(average_acc)

y_onehot_new = keras.utils.to_categorical(y, 10)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_onehot_new, epochs=200, batch_size=128)
y_predit = model.predict(X_test)
y_predit1 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_onehot_new, epochs=200, batch_size=128)
y_predit = model.predict(X_test)
y_predit2 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_onehot_new, epochs=200, batch_size=128)
y_predit = model.predict(X_test)
y_predit3 = np.argmax(y_predit,axis=1)

model = Sequential()
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_onehot_new, epochs=200, batch_size=128)
y_predit = model.predict(X_test)
y_predit4 = np.argmax(y_predit,axis=1)

y_predict = []
for i in range(len(y_predit1)):
    y_predict.append(np.bincount([y_predit1[i],y_predit2[i],y_predit3[i],y_predit4[i]]).argmax())
y_predict_1 = np.array(y_predict)

row=y_predict_1.shape[0]
seq=np.array(range(30000,30000+row))
data = pd.DataFrame({"Id":seq,"y":y_predict_1})
data.to_csv('./res.csv',index=False,header=True)
print(y_predict_1)