import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot
import numpy as np

def plot(real_time,Acc_real,Acc,n_rounds,time_stamps,labels,r_labels,name,kind):
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15))
    plt.subplot(221)

    print(len(time_stamps[0]),n_rounds,len(Acc[0]))
    diff=100
    count=0
    for i in range(len(Acc)):
        d=Acc[i][-1]- Acc_real[0][-1]
        if abs(d)<diff:
            diff=abs(d)
            count=i
    print(count)
    for i in range(len(Acc)):
       
        plt.plot(time_stamps[i], Acc[i], label=labels[i], marker='o',markersize=4)
        if i==count:
            plt.annotate(round(Acc[i][-1],3),size=10,xy=(time_stamps[i][-1], Acc[i][-1]),
            xytext=(time_stamps[i][-1]-100, Acc[i][-1]-0.02),arrowprops=dict(arrowstyle='simple',edgecolor='red',facecolor='red'))
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    title='1.1 Accuracy/Time for different '+kind+' values'
    plt.title(title)
    plt.legend()

    plt.subplot(222)
    for i in range(len(time_stamps)):
        plt.plot(time_stamps[i], [x for x in range(n_rounds[i])], label=labels[i], marker='o',markersize=4)
    plt.xlabel("Time(s)")
    plt.ylabel("Rounds")
    title='1.2 Rounds/Time for distributed for different '+kind+' values'
    plt.title(title)
    plt.legend()

    
    cdict = np.random.rand(len(n_rounds),3)
    plt.subplot(223)

    plt.scatter(n_rounds,[i[-1] for i in Acc],marker='o',c=cdict,zorder=2)
    plt.plot(n_rounds,[i[-1] for i in Acc],c='#6da2f8',zorder=1)
    plt.hlines(Acc_real[0][-1],xmin=0, xmax=max(n_rounds),colors='red',linestyles='dashed',label='Max accuracy for centralized training')
    for i in range(len(Acc)):
        plt.annotate(labels[i],size=10,xy=(n_rounds[i]+2,Acc[i][-1] ))
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    title='1.3 Trade-off Total Rounds/ Final Accuracy,\n with each dot being a different '+kind
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
    return