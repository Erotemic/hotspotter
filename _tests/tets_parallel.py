"""
if __name__ == '__main__':
    print('test parallel')
    multiprocessing.freeze_support()
    import numpy as np

    p = multiprocessing.Pool(processes=8)
    data_list = [np.random.rand(1000, 9) for _ in xrange(1000)]
    data = data_list[0]

    def complex_func(data):
        tmp = 0
        for ix in xrange(0, 100):
            _r = np.random.rand(10, 10)
            u1, s1, v1 = np.linalg.svd(_r)
            tmp += s1[0]
        u, s, v = np.linalg.svd(data)
        return s[0] + tmp

    with helpers.Timer('ser'):
        x2 = map(complex_func, data_list)
    with helpers.Timer('par'):
        x1 = p.map(complex_func, data_list)

    '''
    %timeit p.map(numpy.sqrt, x)
    %timeit map(numpy.sqrt, x)
    '''
"""
