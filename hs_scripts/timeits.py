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


