import numpy as np
from numpy import linalg as LA
from sklearn.datasets import make_moons, make_circles, make_classification 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import shuffle
import traceback
import math
import time

#Φ(x)
def f(x,E,e):
    x,E=same_len(x,E)
    n1=np.append(E[0], E[1])
    n2=np.append(x[0], x[1])
    if LA.norm(n1)!=0:
        t1= -e*LA.norm(n1)-np.dot(n2,(n1/LA.norm(n1)))
    else:
        t1=0
    n_sum=np.add(n2,n1)
    t2=LA.norm(n_sum) - (1+e)*LA.norm(n1)
    return max(t1,t2)


def create_dataset(d,kind):
    
    X, y = make_classification(n_samples=d["n_samples"], n_features=d["n_features"],n_informative=d["n_informative"],
        n_redundant=d["n_redundant"], n_repeated=d["n_repeated"],n_classes=d["n_classes"],n_clusters_per_class=d["n_clusters_per_class"],
        weights=d["weights"],flip_y=d["flip_y"],random_state=d["random_state"])

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=int(len(X)/3))
    try:
        np.save("np_arrays/X", X_train)
        np.save("np_arrays/y", y_train)
        np.save("np_arrays/X_test", X_test)
        np.save("np_arrays/y_test", y_test)
        print("A new dataset has been created...")
    except:
        print("An exception occurred while trying to save the dataset...")
    if kind=='balanced':
        np.save("np_arrays/X_bal", X_train)
        np.save("np_arrays/y_bal", y_train)
        np.save("np_arrays/X_test_bal", X_test)
        np.save("np_arrays/y_test_bal", y_test)
    else:
        np.save("np_arrays/X_Unbal", X_train)
        np.save("np_arrays/y_Unbal", y_train)
        np.save("np_arrays/X_test_Unbal", X_test)
        np.save("np_arrays/y_test_Unbal", y_test)
    return

def load_dataset(kind):
    if kind=="balanced":
            print("Balanced len:",len(np.load("np_arrays/X_bal.npy")))
            np.save("np_arrays/X", np.load("np_arrays/X_bal.npy"))
            np.save("np_arrays/y", np.load("np_arrays/y_bal.npy"))
            np.save("np_arrays/X_test", np.load("np_arrays/X_test_bal.npy"))
            np.save("np_arrays/y_test", np.load("np_arrays/y_test_bal.npy"))
    else:
        print("Unbalanced len:",len(np.load("np_arrays/X_Unbal.npy")))
        np.save("np_arrays/X", np.load("np_arrays/X_Unbal.npy"))
        np.save("np_arrays/y", np.load("np_arrays/y_Unbal.npy"))
        np.save("np_arrays/X_test", np.load("np_arrays/X_test_Unbal.npy"))
        np.save("np_arrays/y_test", np.load("np_arrays/y_test_Unbal.npy"))
    return

#get chunks to fit
def create_chunks(parts):
    X=np.load("np_arrays/X.npy")
    y=np.load("np_arrays/y.npy")
    split= math.ceil(len(X)/parts)
    for i in range(parts):
        start=split*i
        end=start+split
        if start>=(len(X)-1):
            return "False","False"
        if end>(len(X)-1):
            end=len(X)
        name_x="np_arrays/minibatches/X_"+str(i)
        name_y="np_arrays/minibatches/y_"+str(i)
        np.save(name_x, X[start:end])
        np.save(name_y,y[start:end])
    return split

def pred(E,clf,X_test,y_test):
    clf.coef_=np.asarray([E[0]])
    clf.intercept_=np.asarray(E[1])
    clf.classes_=np.asarray([0,1])
    y_pred = clf.predict(X_test)
    #sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
    acc=metrics.accuracy_score(y_test, y_pred)
    return clf,acc

def check_worker(worker):
    status_l=[w.status for w in worker]
    if status_l.count('finished')>=1:
        while status_l.count('finished')!=len(status_l):
            time.sleep(0.01)
            status_l=[w.status for w in worker]
        return "end"

    if status_l.count('error')>=1:
        for w in worker:
            if w.status=="error":
                err=traceback.format_tb(w.traceback())
                print(err)
                return "err"
    elif status_l.count('lost')>=1:
        for w in worker:
            if w.status=="lost":
                print("lost")
                return "lost"
    
    
    return "ok"

def check_coo(coo):
    if coo.status=="error":
        err=traceback.format_tb(coo.traceback())
        print(err)
        return "err"
    elif coo.status=="lost":
        print("lost")
        return "lost"
    
    timeout=0
    while coo.status!="finished":
        timeout+=1
        time.sleep(0.01)
        if timeout*0.01==100: #assuming that timeout error is 100s
            print("Someting went wrong...")
            return "Someting went wrong..."
    return "ok"

def fill_arrays(rounds,sub_rs,feature_array,result):
    if len(rounds)==0:
        rounds.append(result[1])
    else:
        rounds.append(rounds[-1]+result[1])
    sub_rs.extend(result[2])
    E=result[0][0]
    for i in range(len(feature_array)):
        feature_array[i].append(E[i])
    return rounds,sub_rs,feature_array

def random_assign(n_workers,parts):
    n=[i for i in range(parts)]
    shuffle(n)
    X_names=["X_"+str(i) for i in n]
    y_names=["y_"+str(i) for i in n]
    X_assign=[[] for i in range(n_workers)]
    y_assign=[[] for i in range(n_workers)]
    split= math.ceil(len(n)/n_workers)
    print("Number of minibatch for each worker: ",split)
    for i in range(n_workers):
        start=split*i
        end=start+split
        if end>(len(X_names)-1):
            end=len(X_names)
        X_assign[i]=X_names[start:end]
        y_assign[i]=y_names[start:end]

    np.save("np_arrays/X_assign", X_assign)
    np.save("np_arrays/y_assign", y_assign)
    return 

def get_chunk(name,n):
    X_assign=np.load("np_arrays/X_assign.npy")
    y_assign=np.load("np_arrays/y_assign.npy")
    if n>len(X_assign[name]):
        return "False", "False"
    X_name="np_arrays/minibatches/"+X_assign[name][n]+".npy"
    y_name="np_arrays/minibatches/"+y_assign[name][n]+".npy"
    print("MMMMMMM",len(np.load(X_name)))
    return np.load(X_name), np.load(y_name)


def same_len(x,E):
    if len(x[0])!=len(E[0]):
        dif=len(E[0])-len(x[0])
        if dif>0:
            i=0
            while i<dif:
                i+=1
                x[0]=np.append(x[0],0)
            x[1]=x[1]
        elif dif<0:
            i=0
            while i<abs(dif):
                i+=1
                E[0]=np.append(E[0],0)
            E[1]=E[1]
    return x,E
