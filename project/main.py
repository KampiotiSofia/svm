
import traceback
from client import start_client
from coordinator_file import coordinator
from worker_file import worker_f


def main():

    #create client

    client,w = start_client(4)

    #make a dataset and save training X and y 
    X, y = make_classification(n_samples=1200, n_features=4,n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    np.save("X", X_train)
    np.save("y", y_train)
    np.save("X_test", X_test)
    np.save("y_test", y_test)
#     print("X y: ",X_train,y_train)
    clf = linear_model.SGDClassifier()
    coo,w1,w2,w3,w4=future_start(0,[[0,0],[0,0],[0,0],[0,0]],clf)
    print("coo",coo.result())
    future=[coo,w1,w2,w3,w4]
    while True:
        status_l=[w1.status,w2.status,w3.status,w4.status]
#         print("status: ",status_l)
        if status_l.count('error')==len(status_l):
            print("ERROR:\n",w1.traceback(),"\n with format:\n",traceback.format_tb(w1.traceback()) )
        if status_l.count('finished')!=len(status_l):
            if coo.status=='finished':
                print("coo",coo.result())
                E=coo.result()[0]
                del coo
                coo= client.submit(coordinator,4,E,workers=worker1)

        else:
            print("End of chunks.For now on E will stay the same!")
            status_l=[coo.status,w1.status,w2.status,w3.status,w4.status]
            print(status_l)
            while coo.status!='finished':
                print(coo.status)
                time.sleep(1)
            status_l=[coo.status,w1.status,w2.status,w3.status,w4.status]
            print(status_l)
            print("coo",coo.result())
            break
        # here we will predict
    
    future=[coo,w1,w2,w3,w4]
    for f in future: del f

#propably useless
def future_start(E,Si,clf):
    coo= client.submit(coordinator,4,E,workers=worker1)
    w1=client.submit(worker_f,1,Si[0],clf,workers=worker2)
    w2=client.submit(worker_f,2,Si[1],clf, workers=worker3)
    w3=client.submit(worker_f,3,Si[2],clf, workers=worker4)
    w4=client.submit(worker_f,4,Si[3],clf, workers=worker5)
    return coo,w1,w2,w3,w4