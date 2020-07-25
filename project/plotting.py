import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot
import numpy as np
from operator import add
from matplotlib.ticker import StrMethodFormatter

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

    # plt.subplot(223)

    
    # for i in range(len(Acc)):
    #     plt.plot([x for x in range(n_rounds[i][0])],Acc[i],label=labels[i],marker='o',markersize=3)
    
    # plt.xlabel("Rounds")
    # plt.ylabel("Accuracy")
    # title='1.2 Rounds/Time for distributed for different '+kind+' values'
    # plt.title(title)
    # plt.legend()

    plt.subplot(223)
    
    plt.plot(real_time,Acc_real,label=r_labels)
    plt.annotate(round(Acc_real[-1],3),size=10, xy=(real_time[-1], Acc_real[-1]), xytext=(real_time[-1]-1, Acc_real[-1]-0.02),
        arrowprops=dict(arrowstyle='simple',edgecolor='red',facecolor='red'),
        )
    
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.3 Accuracy/Time centralized')
    plt.legend()
    

    # acc_sel=[]
    # limit_Acc=[]
    # limit_time=[]
    # time_sel=time_stamps[2:5]
    # for i in time_sel:
        
    #     limit_time.append([x for x in i if x <= 150])
    # limit_timeReal=[i for i in real_time[0] if i <= 150]
    # # get e=0,2 e=0,3 e=0,4 
    # acc_sel=Acc[2:5]
    # for i in range(len(acc_sel)):
    #     limit_Acc.append(acc_sel[i][:len(limit_time[i])])

    # limit_time.append(limit_timeReal)
    # limit_Acc.append(Acc_real[0][:len(limit_timeReal)])
    # figure(figsize=(15,15))
    # plt.subplot(221)
    # l=[0.2,0.3,0.4,'centralized']

    # for i in range(len(limit_time)):
    #     plt.plot(limit_time[i], limit_Acc[i], label=l[i], marker='o',markersize=3)
    # plt.xlabel("Time(s)")
    # plt.ylabel("Accuracy")
    # title='1.1 Accuracy/Time for different '+kind+' values limited to ~=100s'
    # plt.title(title)
    # plt.legend()

    # plt.subplot(222)
    # for i in range(len(limit_time)):
    #     plt.plot(limit_time[i], [x for x in range(len(limit_time[i]))], label=l[i], marker='o',markersize=3)
    # plt.xlabel("Rounds")
    # plt.ylabel("Accuracy")
    # title='1.2 Rounds/Accuracy for limit to ~=100s '+kind
    # plt.title(title)
    # plt.legend()
    plt.savefig('B_'+name)
    plt.show()
    return

def plot_workers(l,centr_time,centr_acc):
    t_max=[]
    t_min=[]
    a_max=[]
    a_min=[]
    t=[]
    a=[]
    l=[4,8,12,16,20,24,28,32]
    for i in l:
        name1="np_arrays/total/total_time_mean"+str(i)+".npy"
        name2="np_arrays/total/total_acc_mean"+str(i)+".npy"
        t.append(np.load(name1))
        a.append(np.load(name2))
    t1=[i[0] for i in t]
    t2=[i[1] for i in t]
    a1=[i[0] for i in a]
    a2=[i[1] for i in a]

    t_max,t_min,a_max,a_min=get_max_min(l)

    jtplot.style(theme='grade3')
    print(t_max)
    figure(figsize=(15,15))
    plt.subplot(221)
    plt.plot(l,t1,color='b',label='1st pass', marker='o',markersize=5)
    plt.fill_between(l, [i[0] for i in t_max], [i[0] for i in t_min], color='grey', alpha='0.5')
    plt.plot(l,t2,color='m',label='2nd pass', marker='o',markersize=5)
    plt.fill_between(l,[i[1] for i in t_max], [i[1] for i in t_min], color='grey', alpha='0.5')
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
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel("Number of workers")
    plt.ylabel("Accuracy")
    title='Total accuracy for 1 and 2 passes on the dataset/workers'
    plt.title(title)
    plt.legend()

    plt.savefig('B_Plots/workers_test_mnist')
    plt.show()
    
    return

def plot_speedup(l,centr_time,centr_acc):
    t=[]
    a=[]
    for i in l:
        name1="np_arrays/total/total_time_mean"+str(i)+".npy"
        name2="np_arrays/total/total_acc_mean"+str(i)+".npy"
        t.append(np.load(name1))
        a.append(np.load(name2))
    t1=[i[0] for i in t]
    t2=[i[1] for i in t]
    a2=[i[1] for i in a]
    
    total_centr_time=centr_time[0]+centr_time[1]
    total_time=list( map(add, t1, t2) )
    speedup=[total_centr_time/x for x in total_time]
    
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15))
    plt.subplot(221)
    plt.plot(l,speedup,color='b')
    plt.xlabel("Number of workers")
    plt.ylabel("Speed up - centralized/distributed time")
    title='Speed-up for 2 passes on the dataset/workers'
    plt.title(title)
    plt.legend()

    plt.subplot(222)
    plt.ylim(0.8, 1)
    plt.plot(l,a2,color='b',label='distributed')
    plt.axhline(y=centr_acc[1],color='r',linestyle='-.',label='centralized')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # plt.locator_params(axis='y', nbins=5)
    plt.xlabel("Number of workers")
    plt.ylabel("Accuracy")
    title='Total accuracy for 2 passes on the dataset per worker'
    plt.title(title)
    plt.legend()

    plt.savefig('B_Plots/speedup_test_mnist')
    plt.show()
    
    return

def common_range(data):
    sort_data = np.sort(data)
    Q1 = np.percentile(sort_data, 25, interpolation = 'midpoint')  
    Q2 = np.percentile(sort_data, 50, interpolation = 'midpoint')  
    Q3 = np.percentile(sort_data, 75, interpolation = 'midpoint')
    IQR = Q3 - Q1 
    low_lim = Q1 - 1.5 * IQR 
    up_lim = Q3 + 1.5 * IQR 
    outlier =[] 
    data_l=list(data)
    for x in data_l: 
        if ((x> up_lim) or (x<low_lim)):
            outlier.append(x)
    # if len(outlier)<=5:
    #     for x in outlier:
            data_l.remove(x)
    data=np.array(data_l)
    return max(data),min(data)

def get_max_min(l):
    t_max=[]
    t_min=[]
    a_max=[]
    a_min=[]
    for i in l:
        name1="np_arrays/total/total_time"+str(i)+".npy"
        name2="np_arrays/total/total_acc"+str(i)+".npy"
        t=np.load(name1)
        a=np.load(name2)
        t1_M,t1_m=common_range([i[0] for i in t])
        t2_M,t2_m=common_range([i[1] for i in t])
        a1_M,a1_m=common_range([i[0] for i in a])
        a2_M,a2_m=common_range([i[1] for i in a])
        t_max.append([t1_M,t2_M])
        t_min.append([t1_m,t2_m])
        a_max.append([a1_M,a2_M])
        a_min.append([a1_m,a2_m])
    return t_max,t_min,a_max,a_min