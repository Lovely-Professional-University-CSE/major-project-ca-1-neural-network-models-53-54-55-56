#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 02:36:48 2019

@author: abhishek (Frtug)
"""
import numpy as np
import pandas as pd
from neupy import algorithms
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import sys
import os
from  sklearn import  datasets
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tkinter import *
from tkinter import messagebox
import sklearn
from tkinter import *

# pip install pillow 
from PIL import Image, ImageTk


classifier = algorithms.LVQ(n_inputs=4, n_classes=3)
iris=datasets.load_iris()
x=iris.data
y=iris.target
data = pd.DataFrame(iris.data)



def Svm():
    
    os.system('python3 Svm.py')
    
def Som():
    
    os.system('python3 iris_SOM.py')
    
def Lvq():
    
    os.system('python3 ProjectLVQ.py')
    

def Perceptron():
    
    os.system('python3 Perceptron.py')
    
def Compare():
    
    iris=datasets.load_iris()
    #print(iris.data)
    X=iris.data[:,[2,3]] #train
    print(X)
    y=iris.target 
    print(y)
    #plt.plot(X)
    #from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    svc = sklearn.svm.SVC()
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('misclassified %d'%(y_test!=y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy %2f'%accuracy_score(y_test,y_pred))
    msg = messagebox.showinfo('Maximum Accuracy SVM',accuracy_score(y_test,y_pred))

def Visual():
    

   # print(x[:5])
   # print(y[:5])
   # print(x.shape)
    # print(y.head())
    
    #print(y.shape)
    colors = (0,0,0)
    
    plt.scatter(data[0],y,c="red",alpha=0.5)
    plt.title('Scatter plot Sepal length and Species')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    plt.scatter(data[1], y, c="yellow", alpha=0.5)
    plt.title('Scatter plot Sepal Width and Species')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    plt.scatter(data[2], y, c="blue", alpha=0.5)
    plt.title('Scatter plot petal length and Species')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    plt.scatter(data[3], y, c='green', alpha=0.5)
    plt.title('Scatter plot petal Width and Species')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def fuc():
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

    classifier.train(x_train,y_train,epochs=100)
    predictions=classifier.predict(x_test)
    print(accuracy_score(y_test,predictions))
    msg = messagebox.showinfo('Message',accuracy_score(y_test,predictions))

    pickle.dump(classifier,open('NEwQuerty.sav','wb'))
    global lvq
    lvq = accuracy_score(y_test,predictions)
 
              
def algorithms():
    print("Hello")
    if (e1.get() and e2.get() and e3.get() and e4.get()!= ""):
        E1 = float(e1.get())
        E2 = float(e2.get())
        E3 = float(e3.get())
        E4 = float(e4.get())
        u_input = np.array([E1,E2,E3,E4]).reshape(1,-1)
        answer_label.config(text="Species Found!!!!",fg='red')
        print(u_input)
        classs = pickle.load(open('NEwQuerty.sav','rb'))
        print("KIS")
        if classs.predict(u_input) == 0:
               msg = messagebox.showinfo('Message','Setosa')
               print('Setosa')
               status_label.config(text="Setosa")
               img1.grid_forget()
               img2.grid_forget()
               img.grid()
               
               
        elif classs.predict(u_input) == 1:
                msg = messagebox.showinfo('Message','Versicolor')
                print('Versicolor')
                status_label.config(text="Versicolor")
                img.grid_forget()
                img2.grid_forget()
                img1.grid()
                
        else:
            
                msg = messagebox.showinfo('Message','virginica')
                print('virginica')
                status_label.config(text="virginica")
                img.grid_forget()
                img1.grid_forget()
                img2.grid()
                
                

    else:
          msg = messagebox.showinfo('Message','Enter the Values First')
          answer_label.config(text="Enter the Length and Width",fg='green')
          status_label.config(text="-----")
          img.grid_forget()
          img1.grid_forget()
          img2.grid_forget()
          
    print("EXIT")
def fff():
  ##########################################################  
    roots.destroy()
    master = Tk()
    
    master.geometry('1000x1000')
    global answer_label
    answer_label =Label(master, text ="---")
    answer_label.grid(row =10, column =0)
    
    Label(master, text='Sepal_length',height=1, background='black', fg='white',font=("Cursive", 20)).grid(row=2,padx=20) 
    Label(master, text='Sepal_width',height=1, background='black', fg='white',font=("Cursive", 20)).grid(row=3,padx=20)
    Label(master, text='Petal_lenth',height=1, background='black', fg='white',font=("Cursive", 20)).grid(row=4,padx=20) 
    Label(master, text='Petal_width',height=1, background='black', fg='white',font=("Cursive", 20)).grid(row=5,padx=20)
    global e1  
    global e2 
    global e3  
    global e4 
    
    e1 = Entry(master,background='grey') 
    e2 = Entry(master,background='grey')
    e3 = Entry(master,background='grey') 
    e4 = Entry(master,background='grey')
    
    e1.grid(row=2, column=1,pady=10) 
    e2.grid(row=3, column=1,pady=10)
    e3.grid(row=4, column=1,pady=10) 
    e4.grid(row=5, column=1,pady=10)
    
    calculate_button =Button(master, text="FIND", command= algorithms)
    calculate_button.grid(row =7, column =0, columnspan =2)
    
    global status_label
    status_label =Label(master, height =5, width =25, bg ="black", fg ="#00FF00", text ="---", wraplength =150)
    status_label.grid(row =8, column =0, columnspan =2)
    
    global img
    load = Image.open("Iris_setosa.jpg")
    render = ImageTk.PhotoImage(load)
    img = Label(image=render,bg='green',height=250)
    img.image = render
    img.grid(row = 11, column=0,sticky=W) 
    #img.grid_forget()
    
    global img1
    load1 = Image.open("Iris_versicolor.jpg")
    render1 = ImageTk.PhotoImage(load1)
    img1 = Label(image=render1,bg='green',height=250)
    img1.image = render1
    img1.grid(row = 11, column=1) 
    #img1.grid_forget()
    
    
    global img2
    load2 = Image.open("Iris_virginica.jpg")
    render2 = ImageTk.PhotoImage(load2)
    img2 = Label(image=render2,bg='green',height=250)
    img2.image = render2
    img2.grid(row = 11, column=2,sticky=E) 
    #img2.grid_forget()
    mainloop() 
##################################################################    
roots = Tk()
t = roots.geometry('1200x900')
print(t)
roots['bg'] = 'grey'
roots.title('Details')
intro = Label(roots, text='Project ',anchor="center", background='black', fg='white', font=("Bold", 70))
intro.grid(sticky=N,padx=20,pady=20,columnspan=2)


n1 = Label(roots, text=' Machine Learning: ', background='black', fg='white', font=("Cursive", 40))
p1 = Label(roots, text=" MODEL ", background='black', fg='white', font=("Cursive", 40))
n1.grid(row=1, sticky=W, column=0, padx=10, pady=20)
p1.grid(row=2, sticky=W, column=0, padx=10, pady=20)

n2 = Label(roots, text=' IRIS DATASET ', background='black', fg='white', font=("Cursive", 40))
p2 = Label(roots, text=" CLASSIFICATION: ", background='black', fg='white', font=("Cursive", 40))
n2.grid(row=1, sticky=E, column=1, padx=10, pady=20)
p2.grid(row=2, sticky=E, column=1, padx=10, pady=20)


#n2.config(state = 'disabled')
#p2.config(state = 'disabled')

n2.grid(row=1, column=1, sticky=E, padx=20)
p2.grid(row=2, column=1, sticky=E, padx=20)

testButton = Button(roots, text='TEST',fg='red',width=7,height=2, command=fff)
testButton.grid(column=2, row=1, padx=10,sticky=E)

trainButton = Button(roots, text='TRAINING',fg='red',width=7,height=2, command=fuc)
trainButton.grid(column=2, row=2, padx=10,sticky=E)

visButton = Button(roots, text='VISUAL',fg='red',width=7,height=2, command=Visual)
visButton.grid(column=3, row=1, padx=10,pady=10,sticky=E)

compareButton = Button(roots, text='COMPARE',fg='red',width=7,height=2, command=Compare)
compareButton.grid(column=3, row=2, padx=10,pady=10,sticky=E)

setosalabel = Label(roots, text='SETOSA', background='black', fg='white', font=("Cursive", 20))
setosalabel.grid(row = 3,column = 0)

load = Image.open("Iris_setosa.jpg")
render = ImageTk.PhotoImage(load)
img = Label(image=render,height = 250,bg='blue')
img.image = render
img.grid(row = 5, column=0)


versilabel = Label(roots, text='VERSICOLOR', background='black', fg='white', font=("aeril", 20))
versilabel.grid(row = 3,column = 1)
load = Image.open("Iris_Versicolor.jpg")
render = ImageTk.PhotoImage(load)
img = Label(image=render,bg='red')
img.image = render
img.grid(row = 5, column=1)  

virglabel = Label(roots, text='VIRGINICA', background='black', fg='white', font=("Bold",20))
virglabel.grid(row = 3,column = 2)

load = Image.open("Iris_virginica.jpg")
render = ImageTk.PhotoImage(load)
img = Label(image=render,bg='green')
img.image = render
img.grid(row = 5, column=2)    


virglabel = Label(roots, text='Lets run the independent code of all the different Models in the above Buttons', background='black', fg='white', font=("Bold",10))
virglabel.grid(row = 10,column = 0)


### buttons individual

testButton = Button(roots, text='Perceptron',fg='green',width=8,height=2, command=Perceptron)
testButton.grid(column=0, row=9, padx=10)

trainButton = Button(roots, text='LVQ',fg='green',width=6,height=2, command=Lvq)
trainButton.grid(column=1, row=9, padx=10)

visButton = Button(roots, text='SVM',fg='green',width=6,height=2, command=Svm)
visButton.grid(column=2, row=9, padx=10,pady=10)

compareButton = Button(roots, text='SOM',fg='green',width=6,height=2, command=Som)
compareButton.grid(column=3, row=9, padx=10,pady=10)

## 
roots.mainloop()

