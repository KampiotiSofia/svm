import numpy as np
from numpy import linalg as LA

#Φ(x)
def f(x,E):
    if len(x)!=len(E):
        dif=len(E)-len(x)
        for i in range(abs(dif)):
            if dif>0:
                x=np.append(x,0)
            else:
                E=np.append(E,0)
    if LA.norm(E)!=0:
        
        t1= -0.01*LA.norm(E)-np.dot(x,(E/LA.norm(E)))
    else:
        t1=0
    n_sum=np.add(x,E)
    t2=LA.norm(n_sum) - (1+0.01)*LA.norm(E)
    return max(t1,t2)

#get chunks to fit
def get_chunk(n):
    X=np.load("X.npy")
    y=np.load("y.npy")
    n=n-1
    n_parts=200
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

