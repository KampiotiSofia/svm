import numpy as np
from numpy import linalg as LA

#Φ(x)
def f(x,E):
    if len(x[0])!=len(E[0]):
        dif=len(E[0])-len(x[0])
        for i in range(abs(dif)):
            if dif>0:
                x[0]=np.append(x[0],0)
                x[1]=x[1]
            else:
                E[0]=np.append(E[0],0)
                E[1]=E[1]
    if LA.norm(E[0])!=0:
        t1= -0.01*LA.norm(E[0])-np.dot(x[0],(E[0]/LA.norm(E[0])))
    else:
        t1=0
    n_sum=np.add(x[0],E[0])
    t2=LA.norm(n_sum) - (1+0.01)*LA.norm(E[0])
    return max(t1,t2)

#get chunks to fit
def get_chunk(n):
    X=np.load("X.npy")
    y=np.load("y.npy")
    n=n-1
    n_parts=len(X)/10
    split= int(len(X)/n_parts)
    start=split*n
    end=start+split
    if start>=(len(X)-1):
        return "False","False"
    if end>(len(X)-1):
        end=len(X)
    X_train = X[start:end]
    y_train = y[start:end]
    return X_train,y_train

