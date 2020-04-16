from dask.distributed import Client , LocalCluster
def start_client(n):
    cluster = LocalCluster(n_workers=n, threads_per_worker=1)
    client = Client(cluster)
    print(client)
    c=cluster.scheduler.workers
    w=[]
    for i in c.items():
        w.append(i[0])
    # worker1= c.items()[0][0]
    # worker2= c.items()[1][0]
    # worker3= c.items()[2][0]
    # worker4= c.items()[3][0]
    # worker5= c.items()[4][0]
    return client,w
