import numpy as np # for performing linear algebric operations

from sklearn.datasets import load_iris # contains iris dataset

data=load_iris() # load the dataset


x = data.data # get data points
w =np.array([np.random.rand(2) for i in range(len(x))]) #generated random weights using numpy

lrate= 0.6 # learning rate
e=1 # number of epoch 
D=[0,0] # Dimension
print('learning rate of this epoch is',lrate);
while(e<=3): # e is epoch
    print('Epoch is',e);
    
    for i in range(4): # number of patterns 4
        for j in range(2): # size of neurons 
             temp=0
             for k in range(4):
                temp = temp + ((w[k,j]-x[i,k])**2)
             D[j]=temp # distance matrix
        #decide winner neurons
        if(D[0]<D[1]):
            J=0
        else:
            J=1
        
        print('winning unit is',J+1)
        print('weight updation ...')
        for m in range(4):
             w[m,J]=w[m,J] + (lrate *(x[i,m]-w[m,J]))
        print('Updated weights',w)
        

    e=e+1
    lrate = 0.5*lrate;
    print(' updated learning rate after ',e,' epoch is',lrate)


