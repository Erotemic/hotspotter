import pyflann
import numpy as np
import sys

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
def __akmeans_iterate(data, clusters, datax2_clusterx_old, MAX_ITERS=500,
                      AVE_UNCHANGED_THRESH=30, AVE_UNCHANGED_WINDOW=10):
    num_data = data.shape[0]
    num_clusters = clusters.shape[0]
    xx2_unchanged = np.zeros(AVE_UNCHANGED_WINDOW, dtype=np.int32) + AVE_UNCHANGED_THRESH*10
    print('Printing akmeans info in format: (iterx, mean(#unchanged), #unchanged)')
    for xx in xrange(0, MAX_ITERS): 
        # 1) Find each datapoints nearest cluster center
        (datax2_clusterx, _dist) = ann_flann_once(clusters, data, 1)
        # 2) Find new cluster datapoints
        datax_sort    = datax2_clusterx.argsort()
        clusterx_sort = datax2_clusterx[datax_sort]
        _L = 0   
        clusterx2_dataLRx = [None for _ in xrange(num_clusters)]
        for _R in xrange(len(datax_sort)+1): # Slide R
            if _R == num_data or clusterx_sort[_L] != clusterx_sort[_R]:
                clusterx2_dataLRx[clusterx_sort[_L]] = (_L, _R)
                _L = _R
        # 3) Compute new cluster centers
        for clusterx, dataLRx in enumerate(clusterx2_dataLRx):
            if dataLRx is None: continue # ON EMPTY CLUSTER
            (_L, _R) = dataLRx
            clusters[clusterx] = np.mean(data[datax_sort[_L:_R]], axis=0)
        # 4) Check for convergence (no change of cluster id)
        num_changed = (datax2_clusterx_old != datax2_clusterx).sum()
        xx2_unchanged[xx % AVE_UNCHANGED_WINDOW] = num_changed
        ave_unchanged = xx2_unchanged.mean()
        sys.stdout.write('  ('+str(xx)+', '+str(num_changed)+', '+str(ave_unchanged)+'), \n')
        if ave_unchanged < AVE_UNCHANGED_THRESH:
            break
        else: # Iterate
            datax2_clusterx_old = datax2_clusterx
            if xx % 5 == 0: 
                sys.stdout.flush()
    print('  * AKMEANS: converged in %d/%d iters' % (xx+1, MAX_ITERS))
    sys.stdout.flush()
    return (datax2_clusterx, clusters)

#@profile
def akmeans(data, num_clusters=1e6, MAX_ITERS=500,
            AVE_UNCHANGED_THRESH=30, AVE_UNCHANGED_WINDOW=10):
    '''Approximiate K-Means (using FLANN)
    Input: data - np.array with rows of data. dtype must be np.float32
    Description: Quickly partitions data into K=num_clusters clusters.
    Cluter centers are randomly assigned to datapoints. 
    Each datapoint is assigned to its approximate nearest cluster center. 
    The cluster centers are recomputed. 
    Repeat until convergence.'''
    print('Running akmeans: data.shape=%r ; num_clusters=%r' % (data.shape, num_clusters))
    print('  * will converge when the average number of label changes is less than %r over a window of %r iterations' % (AVE_UNCHANGED_THRESH, AVE_UNCHANGED_WINDOW))
    # Setup akmeans iterations
    dtype_ = np.float32  # assert data.dtype == float32
    data   = np.array(data, dtype_) 
    num_data = data.shape[0]
    # Initialize to random cluster clusters
    datax_rand = np.arange(0,num_data);
    np.random.shuffle(datax_rand)
    clusterx2_datax  = datax_rand[0:num_clusters] 
    clusters = np.copy(data[clusterx2_datax])
    datax2_clusterx_old = -np.ones(len(data), dtype=np.int32)
    # This function does the work
    (datax2_clusterx, clusters) = __akmeans_iterate(data, clusters, datax2_clusterx_old, MAX_ITERS,AVE_UNCHANGED_THRESH, AVE_UNCHANGED_WINDOW)
    return (datax2_clusterx, clusters)

def plot_clusters(data, datax2_clusterx, clusters):
    # http://www.janeriksolem.net/2012/03/isomap-with-scikit-learn.html
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    print('Doing PCA')
    num_pca_dims = min(3, data.shape[1])
    pca = PCA(copy=True, n_components=num_pca_dims, whiten=False).fit(data)
    pca_data = pca.transform(data)
    pca_clusters = pca.transform(clusters)
    print('...Finished PCA')

    fig = plt.figure(1)
    fig.clf()
    cmap = plt.get_cmap('hsv')
    data_x = pca_data[:,0]
    data_y = pca_data[:,1]
    data_colors = datax2_clusterx
    clus_x = pca_clusters[:,0]
    clus_y = pca_clusters[:,1]
    clus_colors = np.arange(0,len(clusters))
    if num_pca_dims == 2:
        plt.scatter(data_x, data_y, s=20,  c=data_colors, marker='o')
        plt.scatter(clus_x, clus_y, s=500, c=clus_colors, marker='*')
    if num_pca_dims == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        data_z = pca_data[:,1]
        clus_z = pca_clusters[:,1]
        ax.scatter(data_x, data_y, data_z, s=20,  c=data_colors, marker='o')
        ax.scatter(clus_x, clus_y, clus_z, s=500, c=clus_colors, marker='*')
    return fig

import imp
import os
import hotspotter.helpers
imp.reload(sys.modules['hotspotter.helpers'])
from hotspotter.helpers import load_npz, save_npz, checkpath, remove_file, vd, hashstr_md5

def precompute_akmeans(data, num_clusters=1e6, MAX_ITERS=200, force_recomp=False):
    'precompute akmeans'
    data_md5 = str(data.shape).replace(' ','')+hashstr_md5(data)
    fname = os.path.realpath('precomputed_akmeans_'+data_md5+'.npz')
    if force_recomp:
        remove_file(fname)
    checkpath(fname)
    try: 
        (datax2_clusterx, clusters) = load_npz(fname)
    except Exception as ex:
        (datax2_clusterx, clusters) = akmeans(data, num_clusters)
        save_npz(fname, datax2_clusterx, clusters)
    return (datax2_clusterx, clusters)

if __name__ == '__main__':

    np.random.seed(seed=0) # RANDOM SEED (for reproducibility)
    num_clusters = 10

    __REAL_DATA_MODE__ = True
    if __REAL_DATA_MODE__:
        exec(open('feature_compute2.py').read())
        hs_feats.set_feat_type('HESAFF')
        cx2_desc = hs_feats.cx2_desc
        data = np.vstack(cx2_desc)
        datax2_clusterx, clusters = precompute_akmeans(data, num_clusters, force_recomp=True)
    else:
        data = np.random.rand(1000, 3)
        datax2_clusterx, clusters = akmeans(data, num_clusters)


    fig = plot_clusters(data, datax2_clusterx, clusters)
    fig.show()

    try: 
        __IPYTHON__
    except: 
        plt.show()


#IDEA: 
    #intead have each datapoint "pull" on one another. Maybe warp the space
    #in which they sit with a covariance matrix.  basically let gravity do
    #the clustering.  Check to see if any algos like this. 

    #itertools.groupby
    #groups
