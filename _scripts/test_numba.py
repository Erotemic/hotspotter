from numba import autojit
import numpy as np
import util

@autojit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

def sum2d2(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

def sum2d3(arr):
    M, N = arr.shape
    result = 0.0
    for i in xrange(M):
        for j in xrange(N):
            result += arr[i,j]
    return result

def sum2d4(arr):
    M, N = arr.shape
    result = 0.0
    result = arr.sum()
    return result

@autojit
def sum2d5(arr):
    M, N = arr.shape
    result = 0.0
    result = arr.sum()
    return result

if __name__ == '__main__':
    input_sizes = [1, 10, 100,1000,10000]
    for sz in input_sizes:
        print('\n--------------------\nInput Size: '+str(sz))
        arr = np.random.rand(sz, sz)
        with util.Timer('with numba'):
            res = sum2d(arr)
            print res
        with util.Timer('with numpy'):
            res2 = sum2d4(arr)
            print res2
        with util.Timer('without numpy and numba'):
            res2 = sum2d5(arr)
            print res2
        '''
        with util.Timer('without numba (but smart)'):
            res3 = sum2d3(arr)
            print res3
        with util.Timer('without numba (naive)'):
            res2 = sum2d2(arr)
            print res2
        '''
        print('=================')
