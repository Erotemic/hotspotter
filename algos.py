import drawing_functions2 as df2
from PIL import Image, ImageOps
from numpy import asarray, percentile, uint8, uint16
from os.path import realpath
import helpers
import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import params
import pyflann
import sklearn.decomposition
import sklearn.preprocessing as sklpreproc
import sys
import sklearn
import scipy as sp
import scipy.sparse as spsparse
#imp.reload(sys.modules['hotspotter.helpers'])
#imp.reload(sys.modules['params'])

def sparse_normalize_rows(csr_mat):
    return sklpreproc.normalize(csr_mat, norm='l2', axis=1, copy=False)

def sparse_multiply_rows(csr_mat, csr_vec):
    'Row-wise multiplication of a sparse matrix by a sparse vector'
    sparse_stack = [row.multiply(csr_vec) for row in csr_mat]
    return spsparse.vstack(sparse_stack, format='csr')

def histeq(pil_img):
    img = asarray(pil_img)
    try:
        from skimage import exposure
        'Local histogram equalization'
        # Equalization
        img_eq_float64 = exposure.equalize_hist(img)
        return Image.fromarray(uint8(np.round(img_eq_float64*255)))
    except Exception as ex:
        from tpl.other import imtools
        print('Scikits not found: %s' % str(ex))
        print('Using fallback histeq')
        return Image.fromarray(imtools.histeq(img)).convert('L')

def tune_flann(data, **kwargs):
    flann = pyflann.FLANN()
    num_data = len(data)
    flann_atkwargs = dict(algorithm='autotuned',
                                 target_precision=.01,
                                 build_weight=0.01,
                                 memory_weight=0.0,
                                 sample_fraction=0.001)
    flann_atkwargs.update(kwargs)

    suffix = repr(flann_atkwargs)
    badchar_list = ',{}\': '
    for badchar in badchar_list:
        suffix = suffix.replace(badchar, '')

    print flann_atkwargs

    tuned_params = flann.build_index(data, **flann_atkwargs)

    helpers.myprint(tuned_params)

    out_file = 'flann_tuned'+suffix
    helpers.write_to(out_file, repr(tunned_params))
    flann.delete_index()
    return tuned_params

def __tune():
    tune_flann(sample_fraction=.03, target_precision=.9, build_weight=.01)
    tune_flann(sample_fraction=.03, target_precision=.8, build_weight=.5)
    tune_flann(sample_fraction=.03, target_precision=.8, build_weight=.9)
    tune_flann(sample_fraction=.03, target_precision=.98, build_weight=.5)
    tune_flann(sample_fraction=.03, target_precision=.95, build_weight=.01)
    tune_flann(sample_fraction=.03, target_precision=.98, build_weight=.9)

    tune_flann(sample_fraction=.3, target_precision=.9, build_weight=.01)
    tune_flann(sample_fraction=.3, target_precision=.8, build_weight=.5)
    tune_flann(sample_fraction=.3, target_precision=.8, build_weight=.9)
    tune_flann(sample_fraction=.3, target_precision=.98, build_weight=.5)
    tune_flann(sample_fraction=.3, target_precision=.95, build_weight=.01)
    tune_flann(sample_fraction=.3, target_precision=.98, build_weight=.9)

    tune_flann(sample_fraction=1, target_precision=.9, build_weight=.01)
    tune_flann(sample_fraction=1, target_precision=.8, build_weight=.5)
    tune_flann(sample_fraction=1, target_precision=.8, build_weight=.9)
    tune_flann(sample_fraction=1, target_precision=.98, build_weight=.5)
    tune_flann(sample_fraction=1, target_precision=.95, build_weight=.01)
    tune_flann(sample_fraction=1, target_precision=.98, build_weight=.9)

#__FLANN_ONCE_PARAMS__ = {
    #'algorithm' : 'kdtree', 
    #'trees'     : 8,
    #'checks'    : 128 }

# Look at /flann/algorithms/dist.h for distance clases

#distance_translation = {"euclidean"        : 1, 
                        #"manhattan"        : 2, 
                        #"minkowski"        : 3,
                        #"max_dist"         : 4,
                        #"hik"              : 5,
                        #"hellinger"        : 6,
                        #"chi_square"       : 7,
                        #"cs"               : 7,
                        #"kullback_leibler" : 8,
                        #"kl"               : 8,
                        #"hamming"          : 9,
                        #"hamming_lut"      : 10,
                        #"hamming_popcnt"   : 11,
                        #"l2_simple"        : 12,}

# MAKE SURE YOU EDIT index.py in pyflann

#flann_algos = {
    #'linear'        : 0,
    #'kdtree'        : 1,
    #'kmeans'        : 2,
    #'composite'     : 3,
    #'kdtree_single' : 4,
    #'hierarchical'  : 5,
    #'lsh'           : 6, # locality sensitive hashing
    #'kdtree_cuda'   : 7, 
    #'saved'         : 254, # dont use
    #'autotuned'     : 255,
#}

#multikey_dists = {
    ## Huristic distances
    #('euclidian', 'l2')        :  1,
    #('manhattan', 'l1')        :  2,
    #('minkowski', 'lp')        :  3, # I guess p is the order?
    #('max_dist' , 'linf')      :  4,
    #('l2_simple')              : 12, # For low dimensional points
    #('hellinger')              :  6,
    ## Nonparametric test statistics
    #('hik','histintersect')    :  5,
    #('chi_square', 'cs')       :  7,
    ## Information-thoery divergences
    #('kullback_leibler', 'kl') :  8,
    #('hamming')                :  9, # xor and bitwise sum
    #('hamming_lut')            : 10, # xor (sums with lookup table ; if no sse2)
    #('hamming_popcnt')         : 11, # population count (number of 1 bits)
#}


 #Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 #bit count of A exclusive XOR'ed with B

#flann_distances = {"euclidean"        : 1, 
                   #"manhattan"        : 2, 
                   #"minkowski"        : 3,
                   #"max_dist"         : 4,
                   #"hik"              : 5,
                   #"hellinger"        : 6,
                   #"chi_square"       : 7,
                   #"cs"               : 7,
                   #"kullback_leibler" : 8,
                   #"kl"               : 8 }

#pyflann.set_distance_type('hellinger', order=0)

def ann_flann_once(dpts, qpts, num_neighbors):
    flann = pyflann.FLANN()
    flann.build_index(dpts, **params.__FLANN_ONCE_PARAMS__)
    checks = params.__FLANN_ONCE_PARAMS__['checks']
    (qx2_dx, qx2_dist) = flann.nn_index(qpts, num_neighbors, checks=checks)
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
def akmeans(data, num_clusters=1e6, MAX_ITERS=150,
            AVE_UNCHANGED_THRESH=30, AVE_UNCHANGED_WINDOW=None):
    '''Approximiate K-Means (using FLANN)
    Input: data - np.array with rows of data. dtype must be np.float32
    Description: Quickly partitions data into K=num_clusters clusters.
    Cluter centers are randomly assigned to datapoints. 
    Each datapoint is assigned to its approximate nearest cluster center. 
    The cluster centers are recomputed. 
    Repeat until convergence.'''

    if AVE_UNCHANGED_WINDOW is None:
        AVE_UNCHANGED_WINDOW = int(len(data) / 500)
    print('Running akmeans: data.shape=%r ; num_clusters=%r' % (data.shape, num_clusters))
    print('  * will converge when the average number of label changes is less than %r over a window of %r iterations' % \
          (AVE_UNCHANGED_THRESH, AVE_UNCHANGED_WINDOW))
    # Setup akmeans iterations
    dtype_ = np.float32  # assert data.dtype == float32
    data   = np.array(data, dtype_) 
    num_data = data.shape[0]
    # Initialize to random cluster clusters
    datax_rand = np.arange(0,num_data);
    np.random.shuffle(datax_rand)
    clusterx2_datax     = datax_rand[0:num_clusters] 
    clusters            = np.copy(data[clusterx2_datax])
    datax2_clusterx_old = -np.ones(len(data), dtype=np.int32)
    # This function does the work
    (datax2_clusterx, clusters) = __akmeans_iterate(data, clusters, datax2_clusterx_old, MAX_ITERS, AVE_UNCHANGED_THRESH, AVE_UNCHANGED_WINDOW)
    return (datax2_clusterx, clusters)

def whiten(data):
    'wrapper around sklearn'
    pca = sklearn.decomposition.PCA(copy=True, n_components=None, whiten=True)
    pca.fit(data)
    data_out = pca.transform(data)
    return data_out
def norm_zero_one(data):
    return (data - data.min()) / (data.max() - data.min())
def scale_to_byte(data):
    return np.array(np.round(norm_zero_one(data) * 255), dtype=np.uint8)

def plot_clusters(data, datax2_clusterx, clusters):
    # http://www.janeriksolem.net/2012/03/isomap-with-scikit-learn.html
    print('Doing PCA')
    num_pca_dims = min(3, data.shape[1])
    pca = sklearn.decomposition.PCA(copy=True, n_components=num_pca_dims, whiten=False).fit(data)
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

def precompute_akmeans(data, num_clusters=1e6, MAX_ITERS=200,
                       force_recomp=False, cache_dir=None):
    'precompute akmeans'
    data_md5 = str(data.shape).replace(' ','')+helpers.hashstr_md5(data)
    fname = 'precomp_akmeans_k%d_%s.npz' % (num_clusters, data_md5)

    if cache_dir is None:
        fpath = realpath(fname)
    else:
        fpath = join(fpath, fname)
    
    if force_recomp:
        helpers.remove_file(fpath)
    helpers.checkpath(fpath)
    try: 
        (datax2_clusterx, clusters) = helpers.load_npz(fname)
    except Exception as ex:
        (datax2_clusterx, clusters) = akmeans(data, num_clusters)
        helpers.save_npz(fname, datax2_clusterx, clusters)
    return (datax2_clusterx, clusters)

if __name__ == '__main__':

    np.random.seed(seed=0) # RANDOM SEED (for reproducibility)
    num_clusters = 10

    __REAL_DATA_MODE__ = True
    if __REAL_DATA_MODE__:
        exec(open('feature_compute2.py').read())
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
