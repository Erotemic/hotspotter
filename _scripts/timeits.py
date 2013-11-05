import timeit
import textwrap
# QUESTION: list allocation 
setup = '''
import numpy as np
def alloc_lists1(num_alloc):
    alloc_data = np.empty(num_alloc, dtype=list)
    for i in xrange(num_alloc): alloc_data[i] = [] 
    return alloc_data

# winner winner... obviously
def alloc_lists2(num_alloc):
    return [[] for _ in xrange(num_alloc)]

def alloc_lists3(num_alloc):
    return [[][:] for _ in xrange(num_alloc)]

def alloc_lists4(num_alloc):
    elist = []
    return [elist[:] for _ in xrange(num_alloc)]
'''

number=100000
print timeit.timeit('alloc_lists1(1000)', setup=setup, number=number)
print timeit.timeit('alloc_lists2(1000)', setup=setup, number=number)
print timeit.timeit('alloc_lists3(1000)', setup=setup, number=number)
print timeit.timeit('alloc_lists4(1000)', setup=setup, number=number)

timeit.timeit(x2, setup=setup, number=number)

# QUESTION: str1 + str2 OR str1 % str2
number=10000
setup = '''
import random
from string import ascii_uppercase
from random import choice, randint
LB = 10
UB = 20
'''

# Test with random lengths
# VERDICT: comparible, probably dominated by rand
x1 = " (''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) ) + 'PS') + ''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) ) "
x2 = " (''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) ) + '%s') % ''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) ) "
timeit.timeit(x1, setup=setup, number=number)
timeit.timeit(x2, setup=setup, number=number)


# Test with constant lengths
# VERDICT: Use s1 + s2
number=10000000
x1 = " 'justsomeweirdstring' + 'someotherweird string' "
x2 = " 'justsomeweirdstring%s' % 'someotherweird string' "
timeit.timeit(x1, setup=setup, number=number)
timeit.timeit(x2, setup=setup, number=number)

# Test with 1 constant lengths
# VERDICT: Use s1 + s2
number=100000
x1 = " 'justsomeweirdstring' + ''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) )"
x2 = " 'justsomeweirdstring%s' % ''.join( choice(ascii_uppercase) for x in xrange(randint(LB,UB)) ) "
timeit.timeit(x1, setup=setup, number=number)
timeit.timeit(x2, setup=setup, number=number)



# Test how bad casting to np.arrays is
setup = textwrap.dedent('''
import numpy as np
x = [np.array([1,2,3,4,5]), 
        np.array([1,352,3,4,5]),
        np.array([1,2,12,4,5]), 
        np.array([1,2,3,4,55])]''')
print timeit.timeit('np.array(x)',setup=setup, number=10000)
print timeit.timeit('pass',setup=setup, number=10000)
print timeit.timeit('x',setup=setup, number=10000)

# TODO: Plot number as a graph

import timeit
setup = '''
import numpy as np
num_words = 10
wx2_axs = [[] for _ in xrange(num_words)]
ax2_wx = [1,2,3,4,5,5,5,5,4,3,2,7,4,2,3,6,3,5,7]
'''

try1 = '''
for ax, wx in enumerate(ax2_wx): 
    wx2_axs[wx].append(ax)
'''

try2 = '''
wx2_append = [axs.append for axs in iter(wx2_axs)]
[wx2_append[wx](ax) for (ax, wx) in enumerate(ax2_wx)]
'''
print(timeit.timeit(try1, setup=setup, number=100000))
print(timeit.timeit(try2, setup=setup, number=100000))

try3 = '''
[wx2_axs[wx].append(ax) for (ax, wx) in enumerate(ax2_wx)]
'''

try4 = '''
map(lambda tup: wx2_axs[tup[1]].append(tup[0]), enumerate(ax2_wx))
'''
print(timeit.timeit(try4, setup=setup, number=100000))


timeit.timeit('ax2_wx[0]', setup=setup, number=10000)

print(timeit.timeit(try3, setup=setup, number=100000))



#-------------
# Test ways of componentwise anding a lists of booleans
import timeit
setup = '''
import numpy as np
a = np.random.rand(1000)
b = np.random.rand(1000)
c = np.random.rand(1000)
out = np.zeros((3,len(a)), dtype=np.bool)
'''

test1 = '''
_inliers1 = [ix for ix, tup in 
            enumerate(zip(a > .5, b > .5, c > .5)) 
            if all(tup)]
'''

# WINNER
test2 = '''
_inliers2, = np.where(np.logical_and(np.logical_and(a > .5, b > .5), c > .5))
'''

test3 = '''
_inliers3, = np.where(np.vstack([a > .5, b > .5, c > .5]).all(0))
'''

test4 = '''
np.greater(a, .5, out[0])
np.greater(b, .5, out[1])
np.greater(c, .5, out[2])
_inliers4, = np.where(out.all(0))
'''

print timeit.timeit(test1, setup=setup, number=10000)
print timeit.timeit(test2, setup=setup, number=10000)
print timeit.timeit(test3, setup=setup, number=10000)
print timeit.timeit(test4, setup=setup, number=10000)

#-------------
# Test ways of componentwise anding a lists of booleans
import timeit
setup = '''
import helpers
func1 = lambda var: str(type(var))
def func2(var): return str(type(var))
'''

test1 = '[func1(val) for val in globals().itervalues()]'
test2 = '[func2(val) for val in globals().itervalues()]'

print timeit.timeit(test1, setup=setup, number=100000)
print timeit.timeit(test2, setup=setup, number=100000)



    seen_add = seen.add
    return np.array([False if fx in seen or seen_add(fx) else True for fx in
iter(list_)])

def flag_duplB(list_):
    seen = set([])
    seen_add = seen.add
    return np.array([(fx in seen or seen_add(fx)) is True for fx in list_])

def flag_duplC(list_):
    seen = set([])
    seen_add = seen.add
    return np.array([(fx in seen or seen_add(fx)) == True for fx in list_])

list_ = [1,2,3,4,5,6,7,3,2,8,4,30,40,600,30]
'''

test1 = 'flag_duplA(list_)'
test2 = 'flag_duplB(list_)'
test3 = 'flag_duplC(list_)'

print timeit.timeit(test1, setup=setup, number=1000000)
print timeit.timeit(test2, setup=setup, number=1000000)
print timeit.timeit(test3, setup=setup, number=1000000)




def imread_timeittesst():
    import timeit
    import cv2
    import numpy as np
    import helpers
    from PIL import Image
    import matplotlib.pyplot as plt
    img_fname = '/media/Store/data/work/HSDB_zebra_with_mothers/images/Nid-06_410--Cid-1.JPG'

    setup = '''
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img_fpath = '/media/Store/data/work/HSDB_zebra_with_mothers/images/Nid-06_410--Cid-1.JPG'
    '''
    with helpers.Timer('cv'):
        img1 = cv2.cvtColor(cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with helpers.Timer('PIL'):
        img2 = np.asarray(Image.open(img_fpath))
    with helpers.Timer('plt'):
        img3 = plt.imread(img_fpath)
    print('img1.shape = %r ' % (img1.shape,))
    print('img2.shape = %r ' % (img2.shape,))
    print('img3.shape = %r ' % (img3.shape,))

    im(img1, 101)
    im(img2, 102)
    im(img3, 103)

    test1 = 'cv2.cvtColor(cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)'
    test2 = 'np.asarray(Image.open(img_fpath))'
    test3 = 'plt.imread(img_fpath)'
    print('PIL: '+str(timeit.timeit(test2, setup=setup, number=100)))
    print('PLT: '+str(timeit.timeit(test3, setup=setup, number=100)))
    print('CV: '+str(timeit.timeit(test1, setup=setup, number=100)))
    # It looks like OpenCV actually has a slight edge


'''



#--------------------------
#Some array calculations. Corresponcds to mc2.spatial_nearest_neighbors
#
# Conclusion: sum() > ().sum() > np.sum()
import timeit
setup_small = '''
from itertools import izip
import numpy as np
nQuery = 3
K = 3
qfx2_xy1 = np.array([(.1, .1), (.2, .2), (.3, .3)])
qfx2_xy2 = np.array([((.1, .1), (.2, .2), (.3, .3)), 
                     ((.1, .1), (.2, .2), (.3, .3)), 
                     ((.1, .1), (.2, .2), (.3, .3))])
                     '''
setup_big = '''
nQuery = 1000
K = 10
qfx2_xy1 = np.random.rand(nQuery, 2)
qfx2_xy2 = np.random.rand(nQuery, K, 2)
'''
setup = setup_small

qfx2_xydist = np.array([
[sum((xy1 - xy2)**2) for xy2 in nn_xys]
for (xy1, nn_xys) in izip(qfx2_xy1, qfx2_xy2)])

test1 = '''qfx2_xydist = np.array([
[((xy1 - xy2)**2).sum() for xy2 in nn_xys]
for (xy1, nn_xys) in izip(qfx2_xy1, qfx2_xy2)])'''

test2 = '''qfx2_xydist = np.array([
[sum((xy1 - xy2)**2) for xy2 in nn_xys]
for (xy1, nn_xys) in izip(qfx2_xy1, qfx2_xy2)])'''

test3 = '''qfx2_xydist = np.array([
[np.sum((xy1 - xy2)**2) for xy2 in nn_xys]
for (xy1, nn_xys) in izip(qfx2_xy1, qfx2_xy2)])'''

test11 = '''qfx2_xydist = [[sum((xy1 - xy2)**2) for xy2 in nn_xys]
                            for (xy1, nn_xys) in izip(qfx2_xy1, qfx2_xy2)]'''

test111 = '''qfx2_xydist = [[sum((xy1 - xy2)**2) for xy2 in nn_xys]
                             for (xy1, nn_xys) in zip(qfx2_xy1, qfx2_xy2)]'''

kwargs = dict(number=100000, setup=setup)

#print('test1(().sum()) = %r' % timeit.timeit(test1, **kwargs))
print('test2(sum())    = %r' % timeit.timeit(test2, **kwargs)) # better
#print('test3(np.sum()) = %r' % timeit.timeit(test3, **kwargs))

print('test11(noarray) = %r' % timeit.timeit(test11, **kwargs)) # better
print('test111(noizip) = %r' % timeit.timeit(test11, **kwargs)) # worse 


test_tile = '''
qfx2_K_xy1 = np.rollaxis(np.tile(qfx2_xy1, (K, 1, 1)), 1)
qfx2_xydist = ((qfx2_K_xy1 - qfx2_xy2)**2).sum(2)
'''


print('test_tile(noizip) = %r' % timeit.timeit(test_tile, **kwargs)) # way way better 

