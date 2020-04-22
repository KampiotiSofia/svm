from dask.distributed import Pub, Sub, TimeoutError
from dask.distributed import get_worker,wait
import numpy as np
import time
import math 

from func import f , get_chunk

"""

Pub-Sub structure

Worker : Worker function is used in multiple future (here 4), each one is assigned in a different 
cluster worker.Workers run till there are now more chunks.There are now results needed from this 
futures so, there is no need to restart them (like we do with the coordinator). Each worker reads 
a different file to update Si. 

"""


def worker_f(name,Si,clf):
    sub_init = Sub('Initialize')
    sub_th = Sub('Theta')
    sub_endr = Sub('EndRound')
    sub_endsub = Sub('EndSubRound')
    pub_incr = Pub('Increment')
    pub_f = Pub('Fs')
    pub_x = Pub('Xs')
    

    # get initial E value from coordinator
    def get_init():    
        try:
            init=sub_init.get(timeout=100)
            return init
        except TimeoutError:
            print('Init aknowlegment not received')
            return []

    #get theta from cordinator   
    def get_th():    
        try:
            th=sub_th.get(timeout=100)
            return th
        except TimeoutError:
            print('Theta aknowlegment not received')
            return -10

    #get aknowlegment for continue or stop the rounds    
    def get_endr():
        try:
            endr=sub_endr.get(timeout=100)
            return endr
        except TimeoutError:
            print('EndofRound aknowlegment not received')
            return -10

    #get aknowlegment for continue or stop the subrounds
    def get_endsub():
        try:
            endsub=sub_endsub.get(timeout=100)
            return endsub
        except TimeoutError:
            print('EndofSubRound aknowlegment not received')
            return -10


    #                       ____Start of worker____

    
    th=0
    w_id= get_worker().name #get worker id
    flag=True 
    E=[[0],0]
    S_prev=[0,0]
    Xi=[]
    
    count_chunks=1
    print("worker",w_id,"started...")
    while flag==True: #while this flag stays true there are chunks
        E=get_init() # get E from coordinator
        if np.array_equal(E[0],np.asarray([0])): #if E=0 compute Xi and return Xi to update E
            X,y=get_chunk(count_chunks) #get_newSi(count_chunks,f_name)
            if type(X)==str and type(y)==str:
                flag=False
                print("NO Chunks,hey")
                break
            clf.partial_fit(X,y,np.unique(([0,1])))
            Si = [clf.coef_[0],clf.intercept_[0]]
            Xi=[clf.coef_[0],clf.intercept_[0]]
            pub_x.put(Xi)
            E=get_init() # get E from coordinator
        #begin of round...

        while get_endr()==1: 
            ci=0
            Xi=[[0],0]
            th=get_th() #get theta

            if th==-10:
                break

            #begin of subround...
            
            while get_endsub()==1:
                count_chunks=count_chunks+1

                zi=f(Xi,E)
                X,y=get_chunk(count_chunks) #get_newSi(count_chunks,f_name)
                if type(X)==str and type(y)==str:
                    flag=False
                    print("NO Chunks")
                    pub_incr.put("no")
                else:

                    S_prev[0]=list(Si[0])
                    S_prev[1]=Si[1]
                    clf.partial_fit(X,y,np.unique([0,1]))
                    coef=clf.coef_[0]
                    interc=clf.intercept_[0]
                    Si[0]=coef
                    Si[1]=interc
                    print(w_id,"Si",Si,"Si_prev",S_prev)
                    Xi=[Si[0]-S_prev[0],Si[1]-S_prev[1]]
                    
                    # print(w_id,"Xi",Xi)
                    c_th=0 

                    if th!=0: #avoid division with 0 if th=0 c_th=0
                        c_th=(f(Xi,E)-zi)/th
                    # print(w_id,"c_th",c_th)
                    ci_new=max(ci,math.floor(c_th))
                    # print(w_id,"ci",ci,"new ci",ci_new)
                    if ci!=ci_new: #if we detect a difference send it to the coordinator
                        ci=ci_new
                        pub_incr.put(ci)
                        print(w_id,"Sended...",ci)

            pub_f.put(f(Xi,E)) 
            print(w_id,"END OF ROUND")
            #end of round...

        time.sleep(4)
        pub_x.put(Xi) # send Xi    
    #time.sleep(3)
    print(w_id,"Ended...")
    return Si



#----------------------------------------------------------------------------------------------
# Useful functions
#----------------------------------------------------------------------------------------------

#make 2 np.array same size    
def make_same(x1,x2):
    if len(x1)!=len(x2):
        dif=len(x2)-len(x1)
        
        for i in range(abs(dif)):
            if dif>0:
                x1=np.append(x1,10)
            else:

                x2=np.append(x2,0)
    return x1,x2