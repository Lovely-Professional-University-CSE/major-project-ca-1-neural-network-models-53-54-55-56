from neupy import algorithms
classifier = algorithms.LVQ(n_inputs=4, n_classes=3)
import pickle

from  sklearn import  datasets
import numpy as np
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x[:5])
print(y[:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

classifier.train(x_train,y_train,epochs=100)
predictions=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
u_input = np.array([[6.4, 3.5, 4.5, 1.2]])
if classifier.predict(u_input) == 0:
	print('Setosa')
elif classifier.predict(u_input) == 1:
	print('Versicolor')
else:
	print('virginica')
pickle.dump(classifier,open('NEwQuerty.sav','wb'))

# for using train model 
import numpy as np
import pickle
classs = pickle.load(open('NEwQuerty.sav','rb'))
u_input = np.array([[6.4, 3.5, 4.5, 1.2]])
if classs.predict(u_input) == 0:
	print('Setosa')
elif classs.predict(u_input) == 1:
	print('Versicolor')
else:
	print('virginica')