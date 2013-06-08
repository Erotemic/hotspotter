import timeit
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



