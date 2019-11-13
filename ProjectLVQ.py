#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:40:37 2019

@author: abhishek
"""
import numpy as np
import pandas as pd
from neupy import algorithms
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

from  sklearn import  datasets
import numpy as np

classifier = algorithms.LVQ(n_inputs=4, n_classes=3)
iris=datasets.load_iris()

x=iris.data
y=iris.target.reshape(-1,1)



print(x[:5])
print(y[:5])
print(x.shape)
# print(y.head())

print(y.shape)
colors = (0,0,0)
data = pd.DataFrame(iris.data)

plt.scatter(data[0],y,c="red",alpha=0.5)
plt.title('Scatter plot Sepal length and Species')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.scatter(data[1], y, c=colors, alpha=0.5)
plt.title('Scatter plot Sepal Width and Species')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.scatter(data[2], y, c=colors, alpha=0.5)
plt.title('Scatter plot petal length and Species')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.scatter(data[3], y, c=colors, alpha=0.5)
plt.title('Scatter plot petal Width and Species')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



# Create plot

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

classifier.train(x_train,y_train,epochs=100)
predictions=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
u_input = np.array([[6.9, 3, 5.1, 1.8]])

if classifier.predict(u_input) == 0:
	print('Setosa')
elif classifier.predict(u_input) == 1:
	print('Versicolor')
else:
	print('virginica')
pickle.dump(classifier,open('NEwQuerty.sav','wb'))