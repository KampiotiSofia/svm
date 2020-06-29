

import numpy as np
import time
import sys
from dask.distributed import Pub, Sub,TimeoutError
from sklearn import linear_model
from sklearn import metrics
from coordinator_file import coordinator
from worker_file import worker_f
from func import create_chunks,create_dataset,load_dataset,pred,check_worker,check_coo,random_assign,get_minibatch

def main(client,w,new,dataset_params,e,chunks,n_minibatch):
    print("-------------------------------------------------------------\n")
    print("Start with num of chunks:",chunks,",e:",e)
    E=None
    Acc=[]
    time_stamps=[]
    total_time=[]
    
    sub_results = Sub('results')
    sub_pass = Sub('passes')
    
    #make a dataset and save training X and y ,give sample number and future number
    if new=='yes':
        create_dataset(dataset_params,chunks,w)

    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy") 
    size=dataset_params["n_samples"]-len(X_test)
    print("Minibatch size:",size/(chunks*n_minibatch))
    clf = linear_model.SGDClassifier(shuffle=False)
    
    
    random_assign(len(w)-1,chunks)
    l=0
    loops=2
    total_acc=[0]*loops
    coo=client.submit(coordinator,loops,clf,e,n_minibatch,w,workers=w[0])
    print("In progress...")
    while coo.status=='pending':
        try:
            results=sub_results.get(timeout=0.001)
            E=results[0]
            time_stamps.append(results[3])
            print(results[1:])
            acc=pred(E,clf,X_test,y_test)
            Acc.append(acc)
            total_acc[l]=acc
        except TimeoutError:
            if len(sub_pass.buffer)!=0:
                time.sleep(1)
                if len(sub_results.buffer)!=0:
                    results=sub_results.get(timeout=0.001)
                    E=results[0]
                    time_stamps.append(results[3])
                    print(results[1:])
                    acc=pred(E,clf,X_test,y_test)
                    Acc.append(acc)
                    total_acc[l]=acc
                print(sub_pass.get(timeout=0.01))
                sub_pass.buffer.clear()
                l+=1
        if l>=loops: break
    total_time,total_rounds,time_l=coo.result()
    print("Total time",total_time)
    del coo
    name1="np_arrays/total/total_time"+str(len(w)-1)
    name2="np_arrays/total/total_acc"+str(len(w)-1)
    np.save(name1,total_time)
    np.save(name2,total_acc)
    return Acc,time_l,total_rounds,total_time

def real_partial(minibatches):
    print("Start...")
    clf = linear_model.SGDClassifier(shuffle=False)
    count_chunks=0
    E=[]
    Acc=[]
    f_acc=[]
    t=[]
    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy")

    # walk through all chunks and train a local model
    for i in range(2):
        s_run_time=time.time()
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
            n_minibatch=0
            batch= get_minibatch(X,y,n_minibatch,minibatches)
            print("chunk",count_chunks)
            while batch!= None:
                print("Minibaches",n_minibatch)
                X_b=batch[0]
                y_b=batch[1]
                clf.partial_fit(X_b,y_b,np.unique(([0,1])))
                n_minibatch+=1
                y_pred = clf.predict(X_test)
                #print("Coef:",clf.coef_[0])
                E.append(clf.coef_[0])
                Acc.append(metrics.accuracy_score(y_test, y_pred))
                sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
                batch= get_minibatch(X,y,n_minibatch,minibatches)
                
            
        t_run_time=time.time()
        t.append(t_run_time-s_run_time)
        print("Ended",i)
        count_chunks=0
        
        f_acc.append(Acc[-1])
    return t,Acc,f_acc




