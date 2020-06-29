import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot
import numpy as np

def plot(real_time,Acc_real,Acc,n_rounds,time_stamps,labels,r_labels,name,kind):
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15))
    plt.subplot(221)

    
    count=0
    print(count)
    for i in range(len(Acc)):
       
        plt.plot(time_stamps[i][0], Acc[i], label=labels[i], marker='o',markersize=3)
        
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    title='1.1 Accuracy/Time for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(222)
    for i in range(len(time_stamps)):
        plt.plot(time_stamps[i][0], [x for x in range(n_rounds[i][0])], label=labels[i],marker='o',markersize=3)
    plt.xlabel("Time(s)")
    plt.ylabel("Rounds")
    title='1.2 Rounds/Time for distributed for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(223)

    
    for i in range(len(Acc)):
        plt.plot([x for x in range(n_rounds[i][0])],Acc[i],label=labels[i],marker='o',markersize=3)
    
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    title='1.2 Rounds/Time for distributed for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(224)
    
    plt.plot(real_time,Acc_real,label=r_labels)
    plt.annotate(round(Acc_real[-1],3),size=10, xy=(real_time[-1], Acc_real[-1]), xytext=(real_time[-1]-1, Acc_real[-1]-0.02),
        arrowprops=dict(arrowstyle='simple',edgecolor='red',facecolor='red'),
        )
    
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.4 Accuracy/Time centralized')
    plt.legend()
    plt.savefig(name)
    plt.show()

    acc_sel=[]
    limit_Acc=[]
    limit_time=[]
    time_sel=time_stamps[2:5]
    for i in time_sel:
        
        limit_time.append([x for x in i if x <= 150])
    limit_timeReal=[i for i in real_time[0] if i <= 150]
    # get e=0,2 e=0,3 e=0,4 
    acc_sel=Acc[2:5]
    for i in range(len(acc_sel)):
        limit_Acc.append(acc_sel[i][:len(limit_time[i])])

    limit_time.append(limit_timeReal)
    limit_Acc.append(Acc_real[0][:len(limit_timeReal)])
    figure(figsize=(15,15))
    plt.subplot(221)
    l=[0.2,0.3,0.4,'centralized']

    for i in range(len(limit_time)):
        plt.plot(limit_time[i], limit_Acc[i], label=l[i], marker='o',markersize=3)
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    title='1.1 Accuracy/Time for different '+kind+' values limited to ~=100s'
    plt.title(title)
    plt.legend()

    plt.subplot(222)
    for i in range(len(limit_time)):
        plt.plot(limit_time[i], [x for x in range(len(limit_time[i]))], label=l[i], marker='o',markersize=3)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    title='1.2 Rounds/Accuracy for limit to ~=100s '+kind
    plt.title(title)
    plt.legend()
    plt.savefig('B_'+name)
    plt.show()
    return

def plot_workers(l,centr_time,centr_acc):
    t=[]
    a=[]
    try:
        for i in l:
            name1="np_arrays/total/total_time"+str(i)+".npy"
            name2="np_arrays/total/total_acc"+str(i)+".npy"
            t.append(np.load(name1))
            a.append(np.load(name2))
        t1=[i[0] for i in t]
        t2=[i[1] for i in t]
        a1=[i[0] for i in a]
        a2=[i[1] for i in a]

        jtplot.style(theme='grade3')
        
        figure(figsize=(15,15))
        plt.subplot(221)
        plt.plot(l,t1,color='b',label='1st pass', marker='o',markersize=5)
        plt.plot(l,t2,color='m',label='2nd pass', marker='o',markersize=5)
        plt.axhline(y=centr_time[0], color='g',linestyle='-.',label='centr 1')
        plt.axhline(y=centr_time[1],color='r',linestyle='-.',label='centr 2')
        plt.xlabel("Number of workers")
        plt.ylabel("Time (s)")
        title='Total time for 1 and 2 passes on the dataset/workers'
        plt.title(title)
        plt.legend()

        plt.subplot(222)
        plt.plot(l,a1,color='b',label='1st pass', marker='o',markersize=5)
        plt.plot(l,a2,color='m', label='2nd pass', marker='o',markersize=5)
        plt.axhline(y=centr_acc[0],color='g',linestyle='-.',label='centr 1')
        plt.axhline(y=centr_acc[1],color='r',linestyle='-.',label='centr 2')
        plt.xlabel("Number of workers")
        plt.ylabel("Accuracy")
        title='Total accuracy for 1 and 2 passes on the dataset/workers'
        plt.title(title)
        plt.legend()

        plt.savefig('B_Plots/workers_bigger')
        plt.show()
    except:
        print("Something went wrong")
    return