import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot


def plot(real_time,Acc_real,Acc,n_rounds,time_stamps,labels,r_labels,name):
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15)) 
    plt.subplot(221)

    print(len(time_stamps[0]),n_rounds,len(Acc[0]))


    for i in range(len(Acc)):
        text_pos=1+i*0.1
        plt.plot(time_stamps[i], Acc[i], label=labels[i], marker='o')
        plt.annotate(Acc[i][-1], xy=(time_stamps[i][-1], Acc[i][-1]), xytext=(time_stamps[i][-1]-text_pos, Acc[i][-1]-1),
            arrowprops=dict(arrowstyle='simple'),
            )
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.1 Accuracy/Time for distributed')
    plt.legend()

    plt.subplot(222)
    for i in range(len(time_stamps)):
        plt.plot(time_stamps[i], [x for x in range(n_rounds[i])], label=labels[i], marker='o')
    plt.xlabel("Time(s)")
    plt.ylabel("Rounds")
    plt.title('1.2 Rounds/Time for distributed')
    plt.legend()

    plt.subplot(223)
    for i in range(len(Acc)):
        plt.plot([x for x in range(n_rounds[i])],Acc[i],label=labels[i], marker='o')
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title('1.3 Rounds/Time for distributed')
    plt.legend()

    plt.subplot(224)
    for i in range(len(Acc_real)):
        text_pos=1+i*0.1
        plt.plot(real_time[i],Acc_real[i],label=r_labels[i])
        plt.annotate(Acc_real[i][-1], xy=(real_time[i][-1], Acc_real[i][-1]), xytext=(real_time[i][-1]-text_pos, Acc_real[i][-1]-1),
            arrowprops=dict(arrowstyle='simple'),
            )
    
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.4 Accuracy/Time centralized')
    plt.legend()
    plt.savefig(name)
    plt.show()
    return