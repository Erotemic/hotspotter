import timeit

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
