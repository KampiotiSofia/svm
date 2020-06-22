from dask.distributed import Pub, Sub, TimeoutError
from dask.distributed import get_worker,wait
import numpy as np
import time

from func import f

"""
Pub-Sub structure

Coordinator: Coordinator is used by a future, in specific worker (worker1).
First send E if E=0,askes workers to compute local drifts in order to update E.
This future ends every time that we get a new E. Starts again till workers have no chunks to read 
"""

def coordinator(n_workers,E,n_rounds,e):
    pub_init = Pub('Initialize')
    pub_th = Pub('Theta')
    pub_endr = Pub('EndRound')
    pub_endsub = Pub('EndSubRound')
    #pub_ask_state = Pub('AskState')
    sub_incr = Sub('Increment')
    sub_f = Sub('Fs')
    sub_x = Sub('Xs')
    
    # get increments from workers
    def get_incr():    
            try:
                incr=sub_incr.get(timeout=5)
                print("Coo Received increments...")
                if incr<0: # works as a flag to let coordinator know that chunks are out
                    print("Coo received notice of chunks ended...")
                return incr
            except TimeoutError:
                return 0
    # get fi's from all workers
    def get_fi(n_workers):  
        fis=[]
        
        # fi=sub_f.get(timeout=5)
        # print("Coo received 1 Fi")
        # fis.append(fi)
        print("try to get fis workers:",n_workers)
        for i in range(n_workers):
            try:
                fi=sub_f.get(timeout=5) 
                print("Coo received",i+1,"fi") 
                fis.append(fi)
            except TimeoutError:
                print('Fis Lost worker/workers num=',len(fis))
                break
        return fis
        

    # get xi's from all workers
    def get_xi(n_workers):  
        drifts=[]
        # xi=sub_x.get(timeout=5)
        # print("Coo received 1 xi")
        # drifts.append(xi)
        print("try to get xi workers:",n_workers)
        for i in range(n_workers):
            try:
                xi=sub_x.get(timeout=6)
                print("Coo received",i+1,"xi") 
                drifts.append(xi)
            except TimeoutError:
                print('Lost worker/workers')
                break
        print("Num of workers",len(drifts))
        return drifts

    def check_subcribers(pub,n_workers):
        print("Check...")
        if n_workers==0:
            print("No workers left")
            return "end"
        
        while len(pub.subscribers)<n_workers: #if not all workers subscribe sleep
                time.sleep(0.01)
        
        print("OK Check")
        return "ok"

    #____________________________Start coordinator_________________________________
    
    n_subs=0
    k=n_workers
    th=0
    fis=0
    drifts=0
    sum_xi=0
    incr=0
    e_y=0.01
    
    print("Coo started ... num_workers",k)
    
    if check_subcribers(pub_init,k)=="end":return None
    if E is None: #if E=0 we need to update E
        while len(pub_init.subscribers)<n_workers: #if not all workers subscribe sleep
            time.sleep(0.01)
        pub_init.put(None)
        print("Warmup...Coo Sended E=0...") 
        drifts=get_xi(n_workers) #get local drifts (Xi's)
        if len(drifts)!=k: k=len(drifts)
        print("Coo received xi's...workers=",k)
        
        sum_xi=add_x(drifts)
        e1=sum_xi[0]/k
        e2=sum_xi[1]/k
        E=[e1,e2]
        #if check_subcribers(pub_init,n_workers)=="end":return None
        pub_init.put(E)
        print("Coo Sended E")
    else:
        #if check_subcribers(pub_init,n_workers)=="end":return None
        pub_init.put(E)
        print("Coo Sended E")
    
    y=k*f([[0],0],E,e)
    barrier=e_y*k*f([[0],0],E,e)
    flag=True #use this flag to finish future if chunks are out

	#start of the round...
    print("START ROUND:",n_rounds," workers ",k)
    while y<=barrier: 
        th=-y/(2*k)

        #if check_subcribers(pub_th,n_workers)=="end":return
        pub_th.put(th) #send theta
        print("Coo Sended theta")
        n_subs+=1
        print("START SUBROUND:",n_subs," workers ",k)
        c=0
        fis=[]
        
		#start of the subround...
        neg=0
        while c<k: 
            
            incr=get_incr() #Get increments
            if incr<0: # works as a flag to let coordinator know that chunks are out
                incr=0
                neg+=1
                k=k-1
                if k==0:
                    flag=False
                    break
                #TAG 1 change removed k<4
                # if k<4: break          
            c=c+incr
			#subrounds ended...
        # if k==1:
        #     flag=False
        
        pub_endsub.put(0) #let workers know that subrounds ended
        print("Coo Sended endofSub... num_workers",k) 
        fis=get_fi(k) #get F(Xi)'s from workers
        
        if len(fis)<k:
            #TAG 2 change
            k=len(fis)-neg
        if len(fis)==0 or k<=0: 
            pub_endsub.put(0)
            pub_endr.put(0)
            return None
        print("Coo Received fi's workers=",k)
        
        y=add_f(fis)
        print("y",y)
        if flag==False: #if false chunks are out end future
            print("Coo Sended endofSub..")
            break
	#rounds ended...

    pub_endr.put(0) #let workers know that rounds ended 
    
    print("Coo Sended endofround... num_workers",k)
    drifts=get_xi(len(fis)) #get local drifts (Xi's)
    #TAG 3 change
    if len(drifts)<k:
        k=len(drifts)-neg
    print("Coo Received xi's workers=",k)
    if len(drifts)==0: return None
    
    sum_xi=add_x(drifts)
    e1=E[0]+(sum_xi[0]/len(drifts))
    e2=E[1]+(sum_xi[1]/len(drifts))
    E=[e1,e2]
    n_rounds+=1
    print("Coo ended...")
    # if flag==False:
    #     time.sleep(1)
    return E,n_rounds,n_subs,k


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
		for i in range(dif):
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