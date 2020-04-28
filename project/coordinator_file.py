from dask.distributed import Pub, Sub, TimeoutError
from dask.distributed import get_worker,wait
import numpy as np
import time

from func import f , get_chunk

"""
Pub-Sub structure

Coordinator: Coordinator is used by a future, in specific worker (worker1).
First send E if E=0,askes workers to compute local drifts in order to update E.
This future ends every time that we get a new E. Starts again till workers have no chunks to read 
"""

def coordinator(n_workers,E,e):
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
                incr=sub_incr.get(timeout=15)
                return incr
            except TimeoutError:
                print('Increment aknowlegment not received')
                return -10

    # get fi's from all workers
    def get_fi(n_workers):  
        fis=[]
        try:
            fi=sub_f.get()
            fis.append(fi)
            for i in range(n_workers-1):
                fi=sub_f.get() 
                fis.append(fi)
            return fis
        except:
            print('Fi aknowlegment not received')
            return -10

    # get xi's from all workers
    def get_xi(n_workers):  
        drifts=[]
        try:
            xi=sub_x.get()
            drifts.append(xi)
            for i in range(n_workers-1):
                xi=sub_x.get() 
                drifts.append(xi)
            return drifts
        except:
            print('Xi aknowlegment not received')
            return -10

    def check_subcribers(pub):
        print("Check...")
        while len(pub.subscribers)!=n_workers: #if not all workers subscribe sleep
                time.sleep(0.01)
        print("Exit check...")
        return

    #____________________________Start coordinator_________________________________
	
    
    check_subcribers(pub_init)
    n_rounds=0
    n_subs=0
    k=n_workers
    th=0
    fis=0
    drifts=0
    sum_xi=0
    incr=0
    e_y=0.01
    
    print("Coo started")
    if np.array_equal(E[0],np.asarray([0])): #if E=0 we need to update E
        pub_init.put(([0],0))
        print("Coo Sended E=0...") 
        drifts=get_xi(k) #get local drifts (Xi's)
        print("Coo received xi's...")
        sum_xi=add_x(drifts)
        e1=sum_xi[0]/n_workers
        e2=sum_xi[1]/n_workers
        E=[e1,e2]
        pub_init.put(E)
        print("Coo Sended E")
    else:
        pub_init.put(E)
        print("Coo Sended E")
    
    # print("E",E)
    y=k*f([[0],0],E,e)
    subs=[]
    flag=True #use this flag to finish future if chunks are out

	#start of the round...

    while y<=e_y*k*f([[0],0],E,e): 
        
        n_rounds+=1
        print("START ROUND:",n_rounds)
        th=-y/(2*k)

        check_subcribers(pub_endr)
        pub_endr.put(1) #let worker know that a new round begins
        print("Coo Sended Startofround..")
        check_subcribers(pub_th)
        pub_th.put(th) #send theta
        print("Coo Sended theta")
        c=0
        fis=[]
        n_subs=0
        
		#start of the subround...

        while c<k: 
            
            n_subs+=1
            print("START SUBROUND:",n_subs)
            
            check_subcribers(pub_endsub)
            pub_endsub.put(1) #let worker know that a new subround begins
            print("Coo Sended StartofSub..")
            incr=get_incr() #Get increments
            print("Coo Received increments...")
            if type(incr)==str: # works as a flag to let coordinator know that chunks are out
                print("Coo received notice of chunks ended...")
                flag=False 
                break

            if incr==-10: #if no increments received
                incr=0
            
            c=c+incr
			#subrounds ended...

        if flag==False: #if false chunks are out end future
            check_subcribers(pub_endsub)
            pub_endsub.put(0)
            print("Coo Sended endofSub..")
            break

        check_subcribers(pub_endsub)
        pub_endsub.put(0) #let workers know that subrounds ended
        print("Coo Sended endofSub..") 
        fis=get_fi(k) #get F(Xi)'s from workers
        print("Coo Received fi's")
        y=add_f(fis)
        print("y",y)
        subs.append(n_subs)
	#rounds ended...

    check_subcribers(pub_endr)
    pub_endr.put(0) #let workers know that rounds ended 
    print("Coo Sended endofround..")
    drifts=get_xi(k) #get local drifts (Xi's)
    print("Coo Received xi's")
    sum_xi=add_x(drifts)
    e1=E[0]+(sum_xi[0]/n_workers)
    e2=E[1]+(sum_xi[1]/n_workers)
    E=[e1,e2]
        
    print("Coo ended...")
    return E,n_rounds,subs


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