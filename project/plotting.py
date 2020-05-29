import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from jupyterthemes import jtplot


def plot(real_time,Acc_real,Acc,n_rounds,time_stamps,threshold,labels):
    jtplot.style(theme='grade3')
    
    figure(figsize=(15,15)) 
    plt.subplot(221)

    print(len(time_stamps[0]),n_rounds,len(Acc[0]))


    for i in range(len(Acc)):

        plt.plot(time_stamps[i], Acc[i], label=threshold[i], marker='o')
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.1 Accuracy/Time for distributed')
    plt.legend()

    plt.subplot(222)
    for i in range(len(time_stamps)):
        plt.plot(time_stamps[i], [x for x in range(n_rounds[i])], label=threshold[i], marker='o')
    plt.xlabel("Time(s)")
    plt.ylabel("Rounds")
    plt.title('1.2 Rounds/Time for distributed')
    plt.legend()

    plt.subplot(223)
    for i in range(len(Acc)):
        plt.plot([x for x in range(n_rounds[i])],Acc[i],label=threshold[i], marker='o')
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title('1.3 Rounds/Time for distributed')
    plt.legend()

    plt.subplot(224)
    for i in range(len(Acc_real)):
        plt.plot(real_time[i],Acc_real[i],label=labels[i])
    
    plt.xlabel("Time(s)")
    plt.ylabel("Accuracy")
    plt.title('1.4 Accuracy/Time centralized')
    plt.legend()
    plt.show()
    return