import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from neupy import algorithms
from sklearn.metrics import accuracy_score
import pickle



data = pd.read_csv('iris.csv')
data.head()
X = data.iloc[:,:5]
Y = data.iloc[:,-1]
X.head()
Y.head()
data.info()
data.describe()
data['Species'].value_counts()
temp = data.drop("Id",axis=1)
g = sns.pairplot(temp, hue='Species',markers="+")
plt.show()
g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()
print(X.shape)
# print(y.head())
print(Y.shape)
Y = preprocessing.LabelEncoder().fit_transform(Y)
classifier = algorithms.LVQ(n_inputs=4, n_classes=3)
from  sklearn import  datasets
import numpy as np
iris=datasets.load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

classifier.train(x_train,y_train,epochs=100)
predictions=classifier.predict(x_test)
print(accuracy_score(y_test,predictions))
u_input = np.array([[6.4, 3.5, 4.5, 1.2]])
if classifier.predict(u_input) == 0:
	print('Setosa')
elif classifier.predict(u_input) == 1:
	print('Versicolor')
else:
	print('virginica')

pickle.dump(classifier,open('NEwQuerty.sav','wb'))
