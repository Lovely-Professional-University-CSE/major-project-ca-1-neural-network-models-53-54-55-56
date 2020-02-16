#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:39:13 2019

@author: abhishek
"""
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
iris=datasets.load_iris()
#print(iris.data)
X=iris.data[:,[2,3]] #train
print(X)
y=iris.target 
print(y)
plt.plot(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
svc = sklearn.svm.SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('misclassified %d'%(y_test!=y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy %2f'%accuracy_score(y_test,y_pred))

