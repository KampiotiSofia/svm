from dask.distributed import Client , LocalCluster
def start_client(n):
    cluster = LocalCluster(n_workers=n, threads_per_worker=1)
    client = Client(cluster)
    print(client)
    c=cluster.scheduler.workers
    w=[]
    for i in c.items():
        w.append(i[0])
    return client,w
