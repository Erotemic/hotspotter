import pyflann
import numpy as np

__FLANN_PARAMS__ = {
    'algorithm' : 'kdtree', 
    'trees'     : 8,
    'checks'    : 128 }

def ann_flann_once(dpts, qpts, num_neighbors):
    flann_ = pyflann.FLANN()
    checks = __FLANN_PARAMS__['checks']
    flann_.build_index(dpts, **__FLANN_PARAMS__)
    (qx2_dx, qx2_dist) = flann_.nn_index(qpts, num_neighbors, checks=checks)
    return (qx2_dx, qx2_dist)

#@profile
def akmeans(data, num_clusters=1e6, MAX_ITERS=1000):
    ''' Approximiate Kmeans (using FLANN)
    Input: data - np.array with rows of data. dtype must be np.float32
    Description: Quickly partitions data into K=num_clusters clusters.
    The nearest neighbors of datapoints are approximated using FLANN Cluster
    cluster-ids are randomly initialized Each point then finds its nearest
    cluster cluster Those points now belongs to those clusters.  The cluster of
    the each cluster is computed.  New cluster clusters are assigned.  The
    process is repeated.  '''
    print('Running akmeans: data.shape=%r ; num_clusters=%r' % (data.shape, num_clusters))
    dtype_ = np.float32  # assert data.dtype == float32
    data   = np.array(data, dtype_) 
    num_data = data.shape[0]
    # Initialize to random cluster clusters
    datax_rand = np.arange(0,num_data);
    np.random.shuffle(datax_rand)
    clusterx2_datax  = datax_rand[0:num_clusters] 
    clusters = np.copy(data[clusterx2_datax])
    datax2_clusterx_old = None
    # BEGIN ITERATIONS 
    # Keep updating point assignments. BREAK if MAX_ITERS or assignments uchanged
    for xx in xrange(0, MAX_ITERS): 
        # Find each datapoints (data) nearest cluster (clusters)
        # Assign clusters to datapoints
        sys.stdout.write('.')
        sys.stdout.flush()
        (datax2_clusterx, _dist) = ann_flann_once(clusters, data, 1)
        # Check for convergence (no change of cluster id)
        if np.array_equal(datax2_clusterx, datax2_clusterx_old): 
            print('  * AKMEANS: converged in %d/%d iters' % (xx, MAX_ITERS))
            break
        # Find new cluster datapoints
        datax_sort   = datax2_clusterx.argsort()
        clusterx_sort = datax2_clusterx[datax_sort]
        _L = 0   
        clusterx2_dataLRx = [None for _ in xrange(num_clusters)]
        for _R in xrange(len(datax_sort)+1): # Slide R
            if _R == num_data or clusterx_sort[_L] != clusterx_sort[_R]:
                clusterx2_dataLRx[clusterx_sort[_L]] = (_L, _R)
                _L = _R
        # Compute new cluster centers
        for clusterx, dataLRx in enumerate(clusterx2_dataLRx):
            if dataLRx is None: # ON EMPTY CLUSTER
                continue #clusters[clusterx,:] = 
            (_L, _R) = dataLRx
            clusters[clusterx] = np.mean(data[datax_sort[_L:_R]], axis=0)
        datax2_clusterx_old = datax2_clusterx
        #itertools.groupby
        #groups
    return datax2_clusterx, clusters

def plot_clusters(data, datax2_clusterx, clusters):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    num_pca_dims = min(2, data.shape[1])
    pca = PCA(copy=True, n_components=num_pca_dims, whiten=False).fit(data)
    pca_data = pca.transform(data)
    pca_clusters = pca.transform(clusters)

    fig = plt.figure(1)
    fig.clf()
    cmap = plt.get_cmap('hsv')
    if num_pca_dims == 2:
        plt.scatter(pca_data[:,0], pca_data[:,1], s=20, c=datax2_clusterx)
        plt.scatter(pca_clusters[:,0], pca_clusters[:,1], s=500, c=np.arange(0,len(clusters)), marker='*')
    if num_pca_dims == 3:
        plt.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2], s=50, c=datax2_clusterx)
        plt.scatter(pca_clusters[:,0], pca_clusters[:,1], clusters[:,2], s=500, c=np.arange(0,len(clusters)), marker='*')
    return fig

if __name__ == '__main__':
    np.random.seed(seed=0) # RANDOM SEED (for reproducibility)
    num_clusters = 10

    __DEV_MODE__ = True
    if __DEV_MODE__:
        exec(open('feature_compute2.py').read())
        hs_feats.set_feat_type('HESAFF')
        cx2_desc = hs_feats.cx2_desc
        data = np.vstack(cx2_desc)
    else:
        data = np.random.rand(1000, 2)

    datax2_clusterx, clusters = akmeans(data, num_clusters)
    fig = plot_clusters(data, datax2_clusterx, clusters)
    fig.show()


#IDEA: 
    #intead have each datapoint "pull" on one another. Maybe warp the space
    #in which they sit with a covariance matrix.  basically let gravity do
    #the clustering.  Check to see if any algos like this. 
