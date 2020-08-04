from dask.distributed import Pub, Sub, TimeoutError
from dask.distributed import get_worker,wait
import numpy as np
import pandas as pd
import time
import math 

from func import f, get_minibatch, load_chunks,load_np

"""

Pub-Sub structure

Worker : Worker function is used in multiple future (here 4), each one is assigned in a different 
cluster worker.Workers run till there are now more chunks.There are now results needed from this 
futures so, there is no need to restart them (like we do with the coordinator). Each worker reads 
a different file to update Si. 

"""


def worker_f(name,clf,parts,e,n_workers):
    sub_init = Sub('Initialize')
    sub_th = Sub('Theta')
    sub_endr = Sub('EndRound')
    sub_endsub = Sub('EndSubRound')
    pub_incr = Pub('Increment')
    pub_f = Pub('Fs')
    pub_x = Pub('Xs')
    

    # get initial E value from coordinator
    def get_init():
        w_id= get_worker().name    
        # try:
        print(w_id,"waits to receive E...")
        init=sub_init.get()
        print(w_id,"Received E")
        return init
        # except TimeoutError:
        #     print(w_id,'Error E not received')
        #     return False

    #get theta from cordinator   
    def get_th():
        w_id= get_worker().name    
        try:
            print(w_id,"waits to receive th...")
            th=sub_th.get(timeout=20)
            print(w_id,"Received theta")
            return th
        except TimeoutError:
            print(w_id,'Theta aknowlegment not received')
            return None

    #get aknowlegment for continue or stop the rounds    
    def get_endr():
        try:
            endr=sub_endr.get(timeout=0.1)
            print(w_id,'End of round received')
            return endr
        except TimeoutError:
            return None

    #get aknowlegment for continue or stop the subrounds
    def get_endsub():
        try:
            endsub=sub_endsub.get(timeout=0)
            print(w_id,'End of subround received')
            return endsub
        except TimeoutError:
            return None
    
    


    #                       ____Start of worker____

    
    th=0
    w_id= get_worker().name #get worker id
    print("worker",w_id,"started...")
    start_time=0
    wait_time=0
    train_time=0
    flag=True 
    E=[[0],0]
    Si=[0,0]
    S_prev=[0,0]
    Xi=[[0],0]

    count_chunks=0
    minibatches=0

    #TAG chunks assigned and load first one
    X_chunk_array,y_chunk_array=load_chunks(name) #get the array with the chunk names assigned to this worker

    X_chunk, y_chunk=load_np(X_chunk_array,y_chunk_array,count_chunks)
    count_chunks+=1
    
    
    while flag==True: #while this flag stays true there are chunks
        
        t2=time.time()
        E=get_init() # get E from coordinator
        if E is False:
            pub_incr.put(-1)   
            return clf
        
        if E is None: #if E=0 compute Xi and return Xi to update E
            t1=time.time()
            #TODO make it prettier
            print(w_id,"Warmup....")
            temp=get_minibatch(X_chunk,y_chunk,minibatches,parts) #get_newSi(count_chunks,f_name)
            
            if temp is None:
                minibatches=0
                load=load_np(X_chunk_array,y_chunk_array,count_chunks)
                if load is None:
                    print(w_id,"End of chunks")
                    flag=False 
                    pub_incr.put(-1)  
                    break
                X_chunk, y_chunk=load
                count_chunks+=1
                temp=get_minibatch(X_chunk,y_chunk,minibatches,parts)
            
            minibatches+=1
            X,y=temp
              
            clf.partial_fit(X,y,np.unique(([0,1])))
            
            Si = [clf.coef_[0],clf.intercept_[0]]
            Xi=[clf.coef_[0],clf.intercept_[0]]
            pub_x.put(Xi)
            print(w_id,"Sended Xi")
            start_time+=time.time()-t1
            t2=time.time()
            E=get_init() # get E from coordinator
            if E is False:
                pub_incr.put(-1)   
                break
            
            
            t1=0
        print(w_id,"Start of round") 
        clf.coef_[0]=E[0]
        clf.intercept_[0]=E[1]
        S_prev[0]= np.array(list(E[0]))
        S_prev[1]=E[1]
        # Xi=[[0],0]
        #begin of round...
        wait_time+=time.time()-t2
        while True:
            t2=time.time()
            th=get_th()
            wait_time+=time.time()-t2
            if th=='End':
                break
            t3=time.time()
            print("HEEEEEEEEEYYYYYYYY th",th)
            ci=0
            
            print(w_id,"Received start of subround")
            
            while get_endsub()==None:
                zi=f(Xi,E,e)
                temp=get_minibatch(X_chunk,y_chunk,minibatches,parts) 
                
                while temp is None:
                    load=load_np(X_chunk_array,y_chunk_array,count_chunks)
                    if load is None:
                        print(w_id,"End of chunks",count_chunks,minibatches)
                        flag=False 
                        break
                    X_chunk, y_chunk=load
                    count_chunks+=1
                    minibatches=0
                    temp=get_minibatch(X_chunk,y_chunk,minibatches,parts)
                if flag==False:
                      
                    break
                else:
                    minibatches+=1
                    X,y=temp
                    clf.partial_fit(X,y,np.unique([0,1]))
                    Si[0]=clf.coef_[0]
                    Si[1]=clf.intercept_[0]
                    Xi=[Si[0]-S_prev[0],Si[1]-S_prev[1]]
                    c_th=0 
                    if th!=0: #avoid division with 0 if th=0 c_th=0
                        c_th=(f(Xi,E,e)-zi)/th
                    ci_new=max(ci,math.floor(c_th))
                    if ci!=ci_new: #if we detect a difference send it to the coordinator
                        incr=ci_new-ci
                        pub_incr.put(incr)
                        ci=ci_new
                        print(w_id,"Sended...",incr)
            pub_f.put(f(Xi,E,e))
            print(w_id,"Sended Fi") 
            print(w_id,"End of subround")
            train_time+=time.time()-t3
            if flag==False:
                break
            
            #end of subround...
            
        pub_x.put(Xi) # send Xi
        print(w_id,"Sended Xi")
            # Xi=[[0],0]
        if flag==False:
            break
    # pub_incr.put(-1)
    while True:
        try:
            df=pd.read_csv('data_workers.csv')
            df = df.append({'n_workers':(n_workers-1),'start_time':start_time, 'wait_time':wait_time,'train_time':train_time},ignore_index=True)
            df.to_csv('data_workers.csv',index=False)
            break
        except:
            continue
    print(w_id,"Ended...")
    return clf


