
import traceback
import numpy as np
import time
import sys
from sklearn.datasets import make_moons, make_circles, make_classification 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from coordinator_file import coordinator
from worker_file import worker_f
from func import get_chunk

def main(client,w,n_samples,n_features,parts,e):
   
    E=[]
    Acc=[]
    rounds=[]
    sub_rs=[]
    worker=[]
    E_array=[[] for i in range(n_features)]
    feature_array=[[] for i in range(n_features)]

    #make a dataset and save training X and y ,give sample number and future number
    X_test,y_test=create_dataset(n_samples,n_features)

    clf = linear_model.SGDClassifier(shuffle=False)
    coo=client.submit(coordinator,len(w)-1,([0],0),e,workers=w[0])

    for i in range(len(w)-1):
        worker.append(client.submit(worker_f,i,clf,parts,e,workers=w[i+1]))


    while True:
        #check if anything unexpected happend to the workers
        time.sleep(1)
        c=check_worker(worker)
        if  c=="ok":
            #workers still running 
            if coo.status=='finished':
                result=coo.result()
                E=result[0]
                del coo
                coo= client.submit(coordinator,len(w)-1,E,e,workers=w[0]) 
                print("coo",result)
                # here we will predict
                clf,acc=pred(E,clf,X_test,y_test)
                Acc.append(acc)
                E_array.append(E[0])
                rounds,sub_rs,feature_array=fill_arrays(rounds,sub_rs,feature_array,result)     
        elif c=="end":
            #no chunks workers ended 
            print("\nEnd of chunks...")
            status_l=[w.status for w in worker]
            print("Coordinator:",coo.status,"...\nWorkers:",status_l)

            if check_coo(coo)=="ok":
                coo= client.submit(coordinator,len(w)-1,E,workers=w[0]) 
                print("coo",result)
                # here we will predict
                clf,acc=pred(E,clf,X_test,y_test)
                Acc.append(acc)
                E_array.append(E[0])
                rounds,sub_rs,feature_array=fill_arrays(rounds,sub_rs,feature_array,result)
                print("Finished with no error...")
            break 
        else:
            return 
    del coo
    for f in worker: del f

    return E_array,feature_array,Acc,rounds,sub_rs

def create_dataset(s,f):
    
    X, y = make_classification(n_samples=s, n_features=f,n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    try:
        np.save("X", X_train)
        np.save("y", y_train)
        np.save("X_test", X_test)
        np.save("y_test", y_test)
    except:
        print("An exception occurred while trying to save the dataset...")
    return X_test,y_test

def pred(E,clf,X_test,y_test):
    clf.coef_=np.asarray([E[0]])
    clf.intercept_=np.asarray(E[1])
    clf.classes_=np.asarray([0,1])
    y_pred = clf.predict(X_test)
    sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
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

def real_partial(parts):
    clf = linear_model.SGDClassifier(shuffle=False)
    count_chunks=0
    E=[]
    Acc=[]
    X_test=np.load("X_test.npy")
    y_test=np.load("y_test.npy")
    while True:
        count_chunks+=1
        X,y=get_chunk(count_chunks,parts) #get_newSi(count_chunks,f_name)
        if type(X)==str and type(y)==str:
            flag=False
            print("NO Chunks...")
            break
        clf.partial_fit(X,y,np.unique(([0,1])))
        y_pred = clf.predict(X_test)
        print("Coef:",clf.coef_[0])
        E.append(clf.coef_[0])
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
    return E,Acc