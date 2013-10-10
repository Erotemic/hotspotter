# ### SNIPIT: Namespace Dict 
namespace_dict = freak_params
for key, val in namespace_dict.iteritems():
    exec(key+' = '+repr(val))
# ### ----
import numpy as np
import timeit

x = np.random.rand(100);
y = np.random.rand(100)

setup = '''
from numpy import array, divide, subtract, multiply, add
import numpy as np
x = %r
y = %r
''' % (x,y)


print timeit.timeit('y = np.random.choice(x,size=4,replace=False)', setup=setup, number=10000)
print timeit.timeit('np.random.shuffle(x); y = x[:4]', setup=setup, number=10000)


print timeit.timeit('x / y', setup=setup)
print timeit.timeit('divide(x, y)' , setup=setup)
print '-----'
print timeit.timeit('x - y', setup=setup)
print timeit.timeit('subtract(x, y)', setup=setup)
print '-----'
print timeit.timeit('x + y', setup=setup)
print timeit.timeit('add(x, y)', setup=setup)
print '-----'
print timeit.timeit('x * y', setup=setup)
print timeit.timeit('multiply(x, y)', setup=setup)
print '-----'


setup = '''
from numpy import array, divide, subtract, multiply, add
import scipy.sparse
import scipy.sparse.linalg
from numpy import linalg
import numpy as np
xyz_norm1 = %r
xyz_norm2 = %r
num_pts = xyz_norm1.shape[1]
Mbynine = np.zeros((2*num_pts,9), dtype=np.float32)
for ilx in xrange(num_pts): # Loop over inliers
    # Concatinate all 2x9 matrices into an Mx9 matrcx
    u2      =     xyz_norm2[0,ilx]
    v2      =     xyz_norm2[1,ilx]
    (d,e,f) =    -xyz_norm1[:,ilx]
    (g,h,i) =  v2*xyz_norm1[:,ilx]
    (j,k,l) =     xyz_norm1[:,ilx]
    (p,q,r) = -u2*xyz_norm1[:,ilx]
    Mbynine[ilx*2:(ilx+1)*2,:]  = np.array(\
        [(0, 0, 0, d, e, f, g, h, i),
            (j, k, l, 0, 0, 0, p, q, r) ] )
MbynineSparse = scipy.sparse.lil_matrix(Mbynine)
''' % (xyz_norm1,xyz_norm2)

stmt1 = '''
(_U, _s, V) = linalg.svd(Mbynine)
'''

stmt2 = '''
(_U, _s, V) = scipy.sparse.linalg.svds(MbynineSparse)
'''

print timeit.timeit(stmt1, setup=setup, number=1000)
print timeit.timeit(stmt2, setup=setup, number=1000)



setup = '''
from numpy import array, divide, subtract, multiply, add
import scipy.sparse
import scipy.sparse.linalg
from numpy import linalg
import numpy as np
B = array([[ 0.82587263, -0.32040576],
       [-0.10060362,  0.23781286]])
C = array([[ 0.47064145,  0.84106429],
       [-0.29378553,  0.36523425]])

zeros_2x1 = np.zeros((2,1))
''' 
print timeit.timeit('np.dot(B,C)', setup=setup, number=100000)
print timeit.timeit('B.dot(C)', setup=setup, number=100000) # This is twice as fast

print timeit.timeit('linalg.pinv(B)', setup=setup, number=10000)
print timeit.timeit('linalg.inv(B)', setup=setup, number=10000) # This also is twice as fast, WERID


# concatinate is faster
print timeit.timeit('np.concatenate((C.dot(linalg.pinv(B)), np.zeros((2,1))),
                    axis=1)', setup=setup, number=50000) 
print timeit.timeit('np.hstack((C.dot(linalg.pinv(B)),np.zeros((2,1))))',
                    setup=setup, number=50000)  

print timeit.timeit('np.concatenate((C.dot(linalg.pinv(B)),zeros_2x1), axis=1)',
                    setup=setup, number=50000) 
print timeit.timeit('np.hstack((C.dot(linalg.pinv(B)),zeros_2x1))', setup=setup,
                    number=50000)  

# pinverse is faster here
print timeit.timeit('np.concatenate((C.dot(linalg.inv(B)),zeros_2x1), axis=1)',
                    setup=setup, number=50000) 
print timeit.timeit('np.hstack((C.dot(linalg.inv(B)),zeros_2x1))', setup=setup,
                    number=50000)  
