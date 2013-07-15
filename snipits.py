# ### SNIPIT: Namespace Dict 
namespace_dict = freak_params
for key, val in namespace_dict.iteritems():
    exec(key+' = '+repr(val))
# ### ----

import timeit

x = np.random.rand(100);
y = np.random.rand(100)

setup = '''
from numpy import array, divide, subtract, multiply, add
import numpy as np
x = %r
y = %r
''' % (x,y)

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
