import sys
import os
import pyflann
import params
import numpy as np
import draw_func2 as df2
import helpers
np.random.seed(5)
# Parameters
tdim   =   2; # Target viewing dimensions
dim    =   2; # Calculation dimension
if len(sys.argv) == 2:
    tdim = int(sys.argv[1])
    dim  = int(sys.argv[1])
K      =   4;
checks = 128;
nQuery =   8;
nData  = 1024; 


# Script
def quick_flann_index(data):
    data_flann = pyflann.FLANN()
    flann_params =  params.VSMANY_FLANN_PARAMS
    checks = flann_params['checks']
    data_flann.build_index(data, **flann_params)
    return data_flann

def reciprocal_nearest_neighbors(query, data, data_flann, checks):
    nQuery, dim = query.shape
    # Assign query features to K nearest database features
    (qfx2_dx, qfx2_dists) = data_flann.nn_index(query, K, checks=checks)
    # Assign those nearest neighbors to K nearest database features
    qx2_nn = data[qfx2_dx]
    qx2_nn.shape = (nQuery*K, dim)
    (_nn2_dx, nn2_dists) = data_flann.nn_index(qx2_nn, K, checks=checks)
    # Get the maximum distance of the reciprocal neighbors
    nn2_dists.shape = (nQuery, K, K)
    qfx2_maxdist = nn2_dists.max(2)
    # Test if nearest neighbor distance is less than reciprocal distance
    isReciprocal = qfx2_dists < qfx2_maxdist
    return qfx2_dx, qfx2_dists, isReciprocal 

data  = np.random.rand(nData,  dim)
query = np.random.rand(nQuery, dim)

nQuery = len(query)
# Find query's Nearest Neighbors in data
data_flann = quick_flann_index(data)
(qfx2_dx, qfx2_dists) = data_flann.nn_index(query, K, checks=checks)
qx2_nn = data[qfx2_dx]
# get k-reciprocal nearest neighbors max distance
qx2_nn.shape = (nQuery*K, dim)
(nn2_dx, nn2_dists) = data_flann.nn_index(qx2_nn, K, checks=checks)
nn2_data = data[nn2_dx] # data's nearest neighbors
nn2_dists.shape = (nQuery, K, K)
qx2_nn.shape = (nQuery, K, dim)
qfx2_maxdist = nn2_dists.max(2)
# A neighbor is a K reciprocal if you are within the 
# max distance of the assigned points K nearest neighbors
isReciprocal = qfx2_dists < qfx2_maxdist
krx2_nn  = qx2_nn[isReciprocal]
krx2_qfx = helpers.tiled_range(nQuery, K)[isReciprocal] 
krx2_query = query[krx2_qfx]


# Enforce viewable dimensionality
if dim != tdim:
    import sklearn.decomposition
    print('Plotting pca.transform dimensionality')
    pca = sklearn.decomposition.PCA(copy=True, n_components=tdim, whiten=False)
    pca.fit(data)
    query_  = pca.transform(query)
    data_   = pca.transform(data)
    nn2_data_ = pca.transform(nn2_data)
    qx2_nn_ = pca.transform(qx2_nn)
    krx2_query_ = pca.transform(krx2_query)
    krx2_nn_ = pca.transform(krx2_nn)
else:
    print('Plotting full dimensionality')
    query_  = (query)
    data_   = (data)
    qx2_nn_ = (qx2_nn)
    krx2_query_ = (krx2_query)
    krx2_nn_ = (krx2_nn)

# Figure and Axis
plt = df2.plt
df2.reset()
fig = plt.figure(1)
if tdim == 2: 
    ax  = fig.add_subplot(111)
elif tdim > 2:
    from mpl_toolkits.mplot3d import Axes3D
    ax  = fig.add_subplot(111, projection='3d')

def plot_points(data, color, marker):
    dataT = data.T
    if len(dataT) == 2:
        ax.plot(dataT[0], dataT[1], color=color, marker=marker, linestyle='None')
    elif len(dataT) == 3:
        ax.scatter(dataT[0], dataT[1], dataT[2], color=color, marker=marker)

def plot_lines(point_pairs, color):
    for pair in point_pairs: 
        dataT = pair.T
        if len(dataT) == 2:
            ax.plot(dataT[0], dataT[1], color=color)
        elif len(dataT) == 3:
            ax.plot(dataT[0], dataT[1], dataT[2], color=color)
            #plt.scatter(dataT[0], dataT[1], dataT[2], s=20, color=color)

# Plot query / data
plot_points(data_, 'b', 'x')
plot_points(query_,'b', 'o')
# Plot KNN
qx2_nn_.shape = (nQuery, K, tdim)
point_pairs = [np.vstack((query_[qx], qx2_nn_[qx,k])) for qx in xrange(nQuery) for k in xrange(K)]
plot_lines(point_pairs, (1, 0, 0, .8))
# Plot NN's KNN
qx2_nn_.shape = (nQuery*K, tdim)
nRes = len(qx2_nn_)
point_pairs3 = [np.vstack((qx2_nn_[nnx], nn2_data_[nnx,k])) for nnx in xrange(nRes) for k in xrange(K)]
plot_lines(point_pairs3, (1, .8, .8, .5))

# Plot KRNN
point_pairs2 = map(np.vstack, zip(krx2_query_, krx2_nn_))
plot_lines(point_pairs2, (0, 1, 0, .9))
df2.update()
# Show
df2.set_figtitle('KRNN=(Green), NN=(Red), NNR=(Pink), dims=%r, K=%r' % (dim, K))
exec(df2.present())
