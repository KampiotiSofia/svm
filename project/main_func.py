

import numpy as np
import time
import sys
from sklearn import linear_model
from sklearn import metrics
from coordinator_file import coordinator
from worker_file import worker_f
from func import get_chunk ,create_chunks,create_dataset,load_dataset,pred,check_worker,check_coo,fill_arrays,random_assign

def main(client,w,new,dataset_params,batches,e,kind):
    print("Start with num of mini-batches:",batches,"and e:",e)
    E=[]
    Acc=[]
    rounds=[]
    sub_rs=[]
    worker=[]
    E_array=[[] for i in range(dataset_params["n_features"])]
    feature_array=[[] for i in range(dataset_params["n_features"])]

    #make a dataset and save training X and y ,give sample number and future number
    if new=='yes':
        create_dataset(dataset_params,kind)
    else:
        load_dataset(kind)

    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy") 

    create_chunks(batches)
    random_assign(len(w)-1,batches)

    clf = linear_model.SGDClassifier(shuffle=False)
    coo=client.submit(coordinator,len(w)-1,([0],0),e,workers=w[0])

    for i in range(len(w)-1):
        worker.append(client.submit(worker_f,i,clf,batches,e,workers=w[i+1]))

    print("In progress...")
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
                # print("coo",result)
                # here we will predict
                clf,acc=pred(E,clf,X_test,y_test)
                Acc.append(acc)
                E_array.append(E[0])
                rounds,sub_rs,feature_array=fill_arrays(rounds,sub_rs,feature_array,result)     
        elif c=="end":
            #no chunks workers ended 
            print("End of chunks...")
            status_l=[w.status for w in worker]
            print("Coordinator:",coo.status,"...\nWorkers:",status_l)

            if check_coo(coo)=="ok":
                #print("coo",result)
                # here we will predict
                clf,acc=pred(E,clf,X_test,y_test)
                Acc.append(acc)
                E_array.append(E[0])
                rounds,sub_rs,feature_array=fill_arrays(rounds,sub_rs,feature_array,result)
                print("Finished with no error...\n\n")
            break 
        else:
            return 
    del coo
    for f in worker: del f

    return E_array,feature_array,Acc,rounds,sub_rs

def real_partial(batches):
    print("Start...")
    clf = linear_model.SGDClassifier(shuffle=False)
    count_chunks=0
    E=[]
    Acc=[]
    X_test=np.load("np_arrays/X_test.npy")
    y_test=np.load("np_arrays/y_test.npy")
    while True:
        count_chunks+=1
        X,y=get_chunk(count_chunks,batches) #get_newSi(count_chunks,f_name)
        if type(X)==str and type(y)==str:
            print("NO Chunks...")
            break
        clf.partial_fit(X,y,np.unique(([0,1])))
        y_pred = clf.predict(X_test)
        #print("Coef:",clf.coef_[0])
        E.append(clf.coef_[0])
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        #sys.stdout.write("Accuracy: %f\n" % (100*metrics.accuracy_score(y_test, y_pred)))
    print("Ended")
    return E,Acc