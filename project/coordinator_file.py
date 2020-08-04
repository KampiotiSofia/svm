from dask.distributed import Pub, Sub, TimeoutError
from dask.distributed import get_worker,wait
import numpy as np
import pandas as pd
import time
from worker_file import worker_f 
from dask.distributed import get_client

from func import f

"""
Pub-Sub structure

Coordinator: Coordinator is used by a future, in specific worker (worker1).
First send E if E=0,askes workers to compute local drifts in order to update E.
This future ends every time that we get a new E. Starts again till workers have no chunks to read 
"""

def coordinator(loops,clf,e,n_minibatch,total_workers):
    pub_pass = Pub('passes')
    pub_results = Pub('results')
    pub_init = Pub('Initialize')
    pub_th = Pub('Theta')
    pub_endr = Pub('EndRound')
    pub_endsub = Pub('EndSubRound')
    sub_incr = Sub('Increment')
    sub_f = Sub('Fs')
    sub_x = Sub('Xs')
    
    # get increments from workers
    def get_incr():    
            try:
                incr=sub_incr.get(timeout=0.01)
                print("Coo Received increments...",incr)
                print("WAIT:",sub_incr.buffer)
                if incr<0: # works as a flag to let coordinator know that chunks are out
                    print("Coo received notice of chunks ended...")
                return incr
            except TimeoutError:
                return 0
    # get fi's from all workers
    def get_fi(n_workers):  
        fis=[]
        print("try to get fis workers:",n_workers)
        for i in range(n_workers):
            try:
                fi=sub_f.get(timeout=0.1) 
                print("Coo received",i+1,"fi") 
                fis.append(fi)
            except TimeoutError:
                print('Fis Lost worker/workers num=',len(fis))
                break
        return fis
        

    # get xi's from all workers
    def get_xi(n_workers):  
        drifts=[]
        print("try to get xi workers:",n_workers)
        for i in range(n_workers):
            try:
                xi=sub_x.get(timeout=0.1)
                print("Coo received",i+1,"xi") 
                drifts.append(xi)
            except TimeoutError:
                print('Lost worker/workers')
                break
        print("Num of workers",len(drifts))
        return drifts

    

    #____________________________Start coordinator_________________________________
    
    
    
    E=None
    start_time=0
    wait_time=0
    process_time_round=0
    process_time_sub=0
    th=0
    fis=0
    drifts=0
    sum_xi=0
    incr=0
    e_y=0.01
    time_l=[]
    total_time=[]
    total_rounds=[]
    clf_results=[clf]*(len(total_workers)-1)
    for l in range(loops):
        t_l=[]
        workers=[]
        time_stamb=0
        n_rounds=0
        print("Coo started ...")
        client= get_client()
        for i in range(len(total_workers)-1):
            workers.append(client.submit(worker_f,i,clf_results[i],n_minibatch,e,workers=total_workers[i+1]))
        
        time.sleep(5)
        flag=True #use this flag to finish future if chunks are out
        total_run_time=time.time()
        n_subs=0
        while flag==True:
            workers_status=[w.status for w in workers]
            k=workers_status.count('pending')
            t1=time.time()
            print("NUMBER OF WORKERS...",k)
            if E is None: #if E=0 we need to update E  
                pub_init.put(None)
                print("Warmup...Coo Sended E=0...") 
                drifts=get_xi(k) #get local drifts (Xi's)
                print("Coo received xi's...workers=",k)
                
                sum_xi=add_x(drifts)
                e1=sum_xi[0]/len(drifts)
                e2=sum_xi[1]/len(drifts)
                E=[e1,e2]
                pub_init.put(E)
                print("Coo Sended E")
            else:
                pub_init.put(E)
                print("Coo Sended E")
            
            start_time+=time.time()-t1

            t2=time.time()
            y=k*f([[0],0],E,e)
            barrier=e_y*k*f([[0],0],E,e)
            
            #start of the round...
            wait_time+=time.time()-t2
            print("START ROUND:",n_rounds," workers ",k)
            while y<=barrier:
                t2=time.time() 
                th=-y/(2*k)

                pub_th.put(th) #send theta
                print("Coo Sended theta")
                n_subs+=1
                print("START SUBROUND:",n_subs," workers ",k)
                c=0
                fis=[]
                
                #start of the subround...
                while c<k: 
                    
                    incr=get_incr() #Get increments
                    if incr<0: # works as a flag to let coordinator know that chunks are out
                        incr=0
                    workers_status=[w.status for w in workers]
                    k=workers_status.count('pending')
                    if k==0:
                        flag=False
                    c=c+incr
                    #subrounds ended...
                pub_endsub.put(0) #let workers know that subrounds ended
                wait_time+=time.time()-t2
                t3=time.time()

                sub_incr.buffer.clear() #clear the buffered messages from previous th
                print("Coo Sended endofSub... num_workers",k)
                workers_status=[w.status for w in workers]
                # k=workers_status.count('pending') 
                fis=get_fi(k) #get F(Xi)'s from workers
                
                if len(fis)==0 and len(list(sub_f.buffer))!=0: 
                    fis=get_fi(len(list(sub_f.buffer)))

                if len(fis)==0:
                    flag=False
                    break
                print("Coo Received fi's workers=",k)
                y=add_f(fis)
                print("y",y)
                process_time_sub+=time.time()-t3
                if flag==False: #if false chunks are out end future
                    print("Coo Sended endofSub..")
                    break
                
            #rounds ended...
            
            t4=time.time()
            pub_endr.put(0) #let workers know that rounds ended 
            
            print("Coo Sended endofround... num_workers",k)
            drifts=get_xi(len(fis)) #get local drifts (Xi's)
            print("len of drifts",len(drifts))
            print("Coo Received xi's workers=",k)
            if len(drifts)==0 and len(list(sub_x.buffer))!=0: 
                drifts=get_xi(len(list(sub_x.buffer)))
            if len(drifts)==0:
                process_time_round+=time.time()-t4
                break

            sum_xi=add_x(drifts)
            e1=E[0]+(sum_xi[0]/len(drifts)) #len(drifts)
            e2=E[1]+(sum_xi[1]/len(drifts)) #len(drifts)
            E=[e1,e2]
            print("E computed")
            n_rounds+=1
            time_stamb=time.time()-total_run_time
            t_l.append(time_stamb)
            pub_results.put([E,n_subs,k,time_stamb])
            process_time_round+=time.time()-t4
            if flag==False:
                break
        
        
        clf_results=[w.result() for w in workers]
        for w in workers: del w
        time_l.append(t_l)
        total_time.append(time_stamb)
        total_rounds.append(n_rounds)
        msg="\n**\n Pass "+str(l)+" completed\n**\n"
        print(msg)
        pub_pass.put(msg)
        time.sleep(5)
    print("Coo ended...")
    df=pd.read_csv('data.csv')
    df = df.append({'n_workers':int(len(total_workers)-1),'rounds':total_rounds[0],'subrounds':n_subs, 'start_time':start_time, 'wait_time':wait_time, 'process_time_sub':process_time_sub, 'process_time_round':process_time_round},ignore_index=True)
    df.to_csv('data.csv',index=False)
    return total_time,total_rounds,time_l

#----------------------------------------------------------------------------------------------
# Useful functions
# add , lenght
#-----------------------------------------------------------------------------------------------

#get the longest list from an np.array to fix any size problems (filling zero's)
def longest(l):
	max_len=0
	for i in l:
		if len(i)>max_len:
			max_len=len(i)
	return max_len
    
#add all coef's received from workers 
def add_coef(array):
	length=longest(array)
	sum_x=[0]*length
	for x in array:
		dif=length-len(x)
		for _ in range(dif):
			x=np.append(x,0)
		sum_x=np.add(sum_x,x)
	return sum_x

#add all xi's received from workers
def add_x(l):
    coef=[]
    for pair in l:
        coef.append(pair[0])
    sum_coef=add_coef(coef)
    sum_interc=sum([pair[1] for pair in l])
    return (sum_coef,sum_interc)

#add all fi's received from workers     
def add_f(array):
	sum_x=0
	for x in array:
		sum_x=sum_x+x
	return sum_x


