print('==============================')
print('____ Running test pyflann ____')
#def __ADD_HOTSPOTTOR_ROOT_TO_PYTHON_PATH__():
    #import os, sys
    ## Find the hotspotter root
    #root_dir = os.path.realpath(os.path.dirname(__file__))
    #while True: 
        #test_path = os.path.join(root_dir, '__HOTSPOTTER_ROOT__')
        #if os.path.exists(test_path):
            #break
        #_new_root = os.path.dirname(root_dir)
        #if _new_root == root_dir:
            #raise Exception('Cannot find hotspotter root')
        #root_dir = _new_root
    #hotspotter_root = root_dir
    ## Append it to the python path so you can import it
    #sys.path.append(hotspotter_root)

#__ADD_HOTSPOTTOR_ROOT_TO_PYTHON_PATH__()

#import hotspotter.tpl.pyflann as pyflann
import pyflann
import cPickle
import numpy as np

'''
build_index(self, pts, **kwargs) method of pyflann.index.FLANN instance
    This builds and internally stores an index to be used for
    future nearest neighbor matchings.  It erases any previously
    stored indexes, so use multiple instances of this class to
    work with multiple stored indices.  Use nn_index(...) to find
    the nearest neighbors in this index.
    
    pts is a 2d numpy array or matrix. All the computation is done
    in float32 type, but pts may be any type that is convertable
    to float32.
'''

#alpha = xrange(0,128)
#pts  = np.random.dirichlet(alpha,size=10000, dtype=np.uint8)
#qpts = np.random.dirichlet(alpha,size=100, dtype=np.uint8)

# Test parameters
nump = 10000
numq = 100
dims = 128
dtype = np.float32

# Create query and database data
print('Create random query and database data')
pts  = np.array(np.random.randint(0,255,(nump,dims)), dtype=dtype)
qpts = np.array(np.random.randint(0,255,(nump,dims)), dtype=dtype)

# Create flann object
print('Create flann object')
flann = pyflann.FLANN()

# Build kd-tree index over the data
print('Build the kd tree')
build_params = flann.build_index(pts)
# Find the closest few points to num_neighbors
print('Find some nearest neighbors')
rindex, rdist = flann.nn_index(qpts, num_neighbors=3)
print rindex, rdist

# Save the data to disk
print('Save the data to the disk')
np.savez('test_pyflann_ptsdata.npz', pts)
npload_pts = np.load('test_pyflann_ptsdata.npz')
pts2 = npload_pts['arr_0']

print('Save and delete the FLANN index')
flann.save_index('test_pyflann_index.flann')
flann.delete_index()

print('Reload the data')
flann2 = pyflann.FLANN()
flann2.load_index('test_pyflann_index.flann',pts2)
rindex2, rdist2 = flann2.nn_index(qpts, num_neighbors=3)

print('Find the same nearest neighbors?')
print rindex2, rdist2

if np.all(rindex == rindex2) and np.all(rdist == rdist2):
    print('...SUCCESS!')
else:
    print('...FAILURE!')
print('\n...done testing pyflann')
print('==============================')
