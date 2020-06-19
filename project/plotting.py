import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot
import numpy as np

def plot(real_time,Acc_real,Acc,n_rounds,time_stamps,labels,r_labels,name,kind):
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15))
    plt.subplot(221)

    diff=100
    count=0
    for i in range(len(Acc)):
        d=Acc[i][-1]- Acc_real[0][-1]
        if abs(d)<diff:
            diff=abs(d)
            count=i
    print(count)
    for i in range(len(Acc)):
       
        plt.plot(time_stamps[i], Acc[i], label=labels[i], marker='o',markersize=3)
        
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    title='1.1 Accuracy/Time for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(222)
    for i in range(len(time_stamps)):
        plt.plot(time_stamps[i], [x for x in range(n_rounds[i])], label=labels[i],marker='o',markersize=3)
    plt.xlabel("Time(s)")
    plt.ylabel("Rounds")
    title='1.2 Rounds/Time for distributed for different '+kind+' values'
    plt.title(title)
    plt.legend()

    
    cdict = np.random.rand(len(n_rounds),3)
    plt.subplot(223)

    
    for i in range(len(Acc)):
        plt.plot([x for x in range(n_rounds[i])],Acc[i],label=labels[i],marker='o',markersize=3)
    
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    title='1.2 Rounds/Time for distributed for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(224)
    for i in range(len(Acc_real)):
        plt.plot(real_time[i],Acc_real[i],label=r_labels[i])
        plt.annotate(round(Acc_real[i][-1],3),size=10, xy=(real_time[i][-1], Acc_real[i][-1]), xytext=(real_time[i][-1]-1, Acc_real[i][-1]-0.02),
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