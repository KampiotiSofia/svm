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

def coordinator(n_workers,E):
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


    #____________________________Start coordinator_________________________________
	
    while len(pub_init.subscribers)!=n_workers: #if not all workers subscribe sleep
        time.sleep(0.01)
    
    counter=0
    count_sub=0
    k=n_workers
    th=0
    fis=0
    drifts=0
    sum_xi=0
    incr=0
    e_y=0.01
    
    print("Coo started")
    if E is 0: #if E=0 we need to update E
        pub_init.put(None)
        print("have send e...") 
        drifts=get_xi(k) #get local drifts (Xi's)
        print("got xi's")
        sum_xi=add_x(drifts)
        E=E+sum_xi
        pub_init.put(E)
    else:
        pub_init.put(E)
    
    print("E",E)
    y=k*f([0],E)
    subs=[]
    flag=True #use this flag to finish future if chunks are out

	#start of the round...

    while y<=e_y*k*f([0],E): 
        
        counter=counter+1
        print("START ROUND:",counter)
        th=-y/(2*k)
        print("th",th)
        
        pub_endr.put(1) #let worker know that a new round begins
        pub_th.put(th) #send theta
        
        c=0
        fis=[]
        count_sub=0
        
		#start of the subround...

        while c<k: 
            
            count_sub=count_sub+1
            print("START SUBROUND:",count_sub)
            
            pub_endsub.put(1) #let worker know that a new subround begins
            incr=get_incr() #Get increments

            if type(incr)==str: # works as a flag to let coordinator know that chunks are out
                print("Coo received notice of chunks ended...")
                flag=False 
                break

            if incr==-10: #if no increments received
                incr=0
            
            c=c+incr
            print("c,incr",c,incr)

			#subrounds ended...

        if flag==False: #if false chunks are out end future
            pub_endsub.put(0)
            break
    
        pub_endsub.put(0) #let workers know that subrounds ended 
        fis=get_fi(k) #get F(Xi)'s from workers
        
        y=add_f(fis)
        print("y",y)

	#rounds ended...

    pub_endr.put(0) #let workers know that rounds ended 
    
    drifts=get_xi(k) #get local drifts (Xi's)
    sum_xi=add_x(drifts)
    E=E+sum_xi
        
    print("Coo ended...")
    return E,counter,count_sub


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
    
#add all xi's received from workers 
def add_x(array):
	
	length=longest(array)
	sum_x=[0]*length
	for x in array:
		dif=length-len(x)
		for i in range(dif):
			x=np.append(x,0)
		sum_x=np.add(sum_x,x)
	return sum_x

#add all fi's received from workers     
def add_f(array):
	sum_x=0
	for x in array:
		sum_x=sum_x+x
	return sum_x