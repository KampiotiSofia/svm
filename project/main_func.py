

import numpy as np
import time
import sys
from sklearn import linear_model
from sklearn import metrics
from coordinator_file import coordinator
from worker_file import worker_f
from func import create_chunks,create_dataset,load_dataset,pred,check_worker,check_coo,random_assign

def main(client,w,new,dataset_params,e,chunks,n_minibatch):
    print("Start with num of chunks:",chunks,"and e:",e)
    E=[]
    Acc=[]
    worker=[]
    time_stamps=[]

    #make a dataset and save training X and y ,give sample number and future number
    if new=='yes':
        create_dataset(dataset_params,chunks,w)
    #TAG REMOVED load_dataset

    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy") 

    clf = linear_model.SGDClassifier(shuffle=False)
    coo=client.submit(coordinator,len(w)-1,None,0,e,workers=w[0])

    for i in range(len(w)-1):
        worker.append(client.submit(worker_f,i,clf,n_minibatch,e,workers=w[i+1]))

    print("In progress...")
    start_time=time.time()
    while True:
        #check if anything unexpected happend to the workers
        time.sleep(1)
        c=check_worker(worker)
        if  c=="ok":
            #workers still running 
            if coo.status=='finished':
                end_time=time.time()
                result=coo.result()
                if result is None:
                    break
                else:
                    time_stamps.append(end_time-start_time)
                    start_time=time.time()
                    E=result[0]
                    n_rounds=result[1]
                    del coo
                    status_l=[i.status for i in worker]
                    n_workers=result[3]
                    coo= client.submit(coordinator,n_workers,E,n_rounds,e,workers=w[0]) 
                    print("coo",result)
                    # here we will predict
                    clf,acc=pred(E,clf,X_test,y_test)
                    Acc.append(acc)   
        elif c=="end":
            #no chunks workers ended 
            
            break 
        else:
            return 
    print("End of chunks...")
    status_l=[w.status for w in worker]
    print("Coordinator:",coo.status,"...\nWorkers:",status_l)

    if check_coo(coo)=="ok":
        # print("coo",result)
        # # here we will predict
        # clf,acc=pred(E,clf,X_test,y_test)
        # Acc.append(acc)
        print("Finished with no error...\n\n")
    del coo
    for f in worker: del f

    return Acc,n_rounds,time_stamps

def real_partial(minibatches):
    print("Start...")
    clf = linear_model.SGDClassifier(shuffle=False)
    count_chunks=0
    E=[]
    Acc=[]
    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy")

    # walk through all chunks and train a local model
    while True:
        if count_chunks>=100:
            print("NO Chunks...")
            break
        name_X="np_arrays/chunks/X_"+str(count_chunks)+".npy"
        name_y="np_arrays/chunks/y_"+str(count_chunks)+".npy"
        X=np.load(name_X)
        y=np.load(name_y)
        count_chunks+=1
        clf.partial_fit(X,y,np.unique(([0,1])))
        y_pred = clf.predict(X_test)
        print("Coef:",clf.coef_[0])
        E.append(clf.coef_[0])
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
    print("Ended")
    return E,Acc