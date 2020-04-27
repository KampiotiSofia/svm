
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


def main(client,w,n_samples,n_features):
    #make a dataset and save training X and y ,give sample number and future number
    X_test,y_test=create_dataset(n_samples,n_features)

    clf = linear_model.SGDClassifier(shuffle=False)
    coo=client.submit(coordinator,len(w)-1,([0],0),workers=w[0])

    E=[]
    
    worker=[]
    for i in range(len(w)-1):
        worker.append(client.submit(worker_f,i,[0,0],clf,workers=w[i+1]))


    while True:
        #check if anything unexpected happend to the workers
        c=check_worker(worker)
        if  c=="ok":
            #workers still running 
            if coo.status=='finished':
                print("coo",coo.result())
                E=coo.result()[0]
                clf=pred(E,clf,X_test,y_test)
                del coo
                coo= client.submit(coordinator,len(w)-1,E,workers=w[0])       
        elif c=="end":
            #no chunks workers ended 
            print("\nEnd of chunks.For now on E will stay the same!")
            status_l=[coo.status,worker[0].status,worker[1].status,worker[2].status,worker[3].status]
            print(status_l)
            if check_coo(coo)=="ok":
                print("coo",coo.result())
                clf=pred(E,clf,X_test,y_test)
                break
            else:
                return
            
        else:
            return
        
        
        # here we will predict
    
    del coo
    for f in worker: del f

    return 

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
    return clf

def check_worker(worker):
    status_l=[worker[0].status,worker[1].status,worker[2].status,worker[3].status]
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
    if status_l.count('finished')>=1:
        while status_l.count('finished')!=len(status_l):
            time.sleep(0.01)
            status_l=[worker[0].status,worker[1].status,worker[2].status,worker[3].status]
        return "end"
    
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