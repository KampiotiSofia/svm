

import numpy as np
import time
import sys
from sklearn import linear_model
from sklearn import metrics
from coordinator_file import coordinator
from worker_file import worker_f
from func import create_chunks,create_dataset,load_dataset,pred,check_worker,check_coo,random_assign

def main(client,w,new,dataset_params,e,chunks,n_minibatch):
    print("-------------------------------------------------------------\n")
    print("Start with num of chunks:",chunks,",e:",e)
    E=None
    Acc=[]
    worker=[]
    time_stamps=[]
    
    #make a dataset and save training X and y ,give sample number and future number
    if new=='yes':
        create_dataset(dataset_params,chunks,w)

    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy") 
    size=dataset_params["n_samples"]-len(X_test)
    print("Minibatch size:",size/(chunks*n_minibatch))
    clf = linear_model.SGDClassifier(shuffle=False)
    clf_results=[clf]*(len(w)-1)
    start_time=time.time()
    start_rounds=0
    for p in range(4):
        random_assign(len(w)-1,chunks)

        for i in range(len(w)-1):
            worker.append(client.submit(worker_f,i,clf_results[i],n_minibatch,e,workers=w[i+1]))
        #TAG Changed the call of random assign + added a for loop + return the clf and feed it again + print after finished
        coo=client.submit(coordinator,len(w)-1,E,start_rounds,e,workers=w[0])
        print("In progress...")
        
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
                        n_workers=result[3]
                        if n_workers==0:
                            break
                        time_stamps.append(end_time-start_time)
                        del coo
                        E=result[0]
                        n_rounds=result[1]
                        status_l=[i.status for i in worker]
                        
                        coo= client.submit(coordinator,n_workers,E,n_rounds,e,workers=w[0])
                        print("coo",result[1:])
                        # here we will predict
                        acc=pred(E,clf,X_test,y_test)
                        Acc.append(acc)   
            elif c=="end":
                if coo.status=='finished':
                    end_time=time.time()
                    result=coo.result()
                    
                    if result is None:
                        break
                    else:
                        #no chunks workers ended
                        E=result[0]
                        n_rounds=result[1]
                        time_stamps.append(end_time-start_time)
                        print("coo",result[1:])
                        # here we will predict
                        acc=pred(E,clf,X_test,y_test)
                        Acc.append(acc)
                        break 
            else:
                break
                #return
        start_rounds=n_rounds
        print("End of chunks...")
        status_l=[w.status for w in worker]
        clf_results=[x.result() for x in worker]
        print("Coordinator:",coo.status,"...\nWorkers:",status_l)
        if check_coo(coo)=="ok":
            print("Finished ",p," pass with no error...\n\n")
        del coo
        for f in worker: del f
        worker=[]
        time.sleep(5)

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
    for i in range(4):
        print("----------------------------------------------\n")
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
            #print("Coef:",clf.coef_[0])
            E.append(clf.coef_[0])
            Acc.append(metrics.accuracy_score(y_test, y_pred))
            sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
        print("Ended",i)
        count_chunks=0
    return E,Acc