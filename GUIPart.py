#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 02:36:48 2019

@author: abhishek
"""
import numpy as np
import pandas as pd
from neupy import algorithms
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

from  sklearn import  datasets
import numpy as np

from sklearn.metrics import accuracy_score

from tkinter import *
from tkinter import messagebox

classifier = algorithms.LVQ(n_inputs=4, n_classes=3)

def fuc():
    iris=datasets.load_iris()
    x=iris.data
    y=iris.target
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

    classifier.train(x_train,y_train,epochs=100)
    predictions=classifier.predict(x_test)
    print(accuracy_score(y_test,predictions))
    msg = messagebox.showinfo('Message',accuracy_score(y_test,predictions))

    pickle.dump(classifier,open('NEwQuerty.sav','wb'))
       
            
def algorithms():
    print("Hello")
    if (e1.get() and e2.get() and e3.get() and e4.get()!= ""):
        E1 = float(e1.get())
        E2 = float(e2.get())
        E3 = float(e3.get())
        E4 = float(e4.get())
        u_input = np.array([E1,E2,E3,E4]).reshape(1,-1)
        print(u_input)
        classs = pickle.load(open('NEwQuerty.sav','rb'))
        print("KIS")
        if classs.predict(u_input) == 0:
               msg = messagebox.showinfo('Message','Setosa')
               print('Setosa')
               status_label.config(text="Setosa")
               
        elif classs.predict(u_input) == 1:
                msg = messagebox.showinfo('Message','Versicolor')
                print('Versicolor')
                status_label.config(text="Versicolor")

        else:
            
                msg = messagebox.showinfo('Message','virginica')
                print('virginica')
                status_label.config(text="virginica")

    else:
          msg = messagebox.showinfo('Message','Enter the Values First')
    print("EXIT")
def fff():
  ##########################################################  
    roots.destroy()
    master = Tk()
    
    master.geometry('400x300')
    answer_label =Label(master, text ="---")
    answer_label.grid(row =10, column =0)
    
    Label(master, text='Sepal_length').grid(row=0) 
    Label(master, text='Sepal_width').grid(row=1)
    Label(master, text='Petal_lenth').grid(row=2) 
    Label(master, text='Petal_width').grid(row=3)
    global e1  
    global e2 
    global e3  
    global e4 
    
    e1 = Entry(master) 
    e2 = Entry(master)
    e3 = Entry(master) 
    e4 = Entry(master)
    
    e1.grid(row=0, column=1) 
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1) 
    e4.grid(row=3, column=1)
    
    calculate_button =Button(master, text="Find", command= algorithms)
    calculate_button.grid(row =7, column =0, columnspan =2)
    
    global status_label
    status_label =Label(master, height =5, width =25, bg ="black", fg ="#00FF00", text ="---", wraplength =150)
    status_label.grid(row =8, column =0, columnspan =2)
    mainloop() 

##################################################################    
roots = Tk()
t = roots.geometry('600x400')
print(t)
roots['bg'] = 'grey'
roots.title('Details')
intro = Label(roots, text='Project details', background='black', fg='white', font=("Bold", 40))
intro.grid(row=0, sticky=N, padx=50, pady=10)

n1 = Label(roots, text=' Machine Learning: ', background='black', fg='white', font=("Cursive", 20))
p1 = Label(roots, text=" Dataset: ", background='black', fg='white', font=("Cursive", 20))
n1.grid(row=1, sticky=W, column=0, padx=20, pady=20)
p1.grid(row=2, sticky=W, column=0, padx=20, pady=20)

n2 = Entry(roots)
p2 = Entry(roots)
n2.insert(0, "a  value")
p2.insert(0, "a value")
n2.config(state = 'disabled')
p2.config(state = 'disabled')

n2.grid(row=1, column=1, sticky=E, padx=20)
p2.grid(row=2, column=1, sticky=E, padx=20)

sigButton = Button(roots, text='TEST', command=fff)
sigButton.grid(column=0, row=3, padx=0,columnspan=1)

sigButton = Button(roots, text='TRAINING', command=fuc)
sigButton.grid(column=0, row=3, padx=0,sticky=E)
roots.mainloop()

