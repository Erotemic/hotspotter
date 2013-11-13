import timeit
import textwrap
import random


def random_cx2_fm(num_cx=10, nmatches_mean=2000):
    def rint(): return random.randint(0,1000)
    def rfloat(): return random.random()*1000
    def rtup(size=2): return tuple([rint() for _ in xrange(size)])
    cx2_fm = []
    cx2_fs = []
    cx2_fk = []
    cx2_numfm = map(lambda _: max(0, _), map(int, np.random.randn(num_cx)*500+nmatches_mean))
    for nfm in cx2_numfm:
        cx2_fm.append([rtup() for _ in xrange(nfm)])
        cx2_fs.append([rfloat() for _ in xrange(nfm)])
        cx2_fk.append([rint() for _ in xrange(nfm)])
    return cx2_fm, cx2_fs, cx2_fk

cx2_fm, cx2_fs, cx2_fk = random_cx2_fm(100)

setup = '''
import numpy as np
FM_DTYPE  = np.uint32
FK_DTYPE  = np.int16
FS_DTYPE  = np.float32
cx2_fm = %r
cx2_fk = %r
cx2_fs = %r
arr_ = np.array
fm_dtype_ = FM_DTYPE
fs_dtype_ = FS_DTYPE
fk_dtype_ = FK_DTYPE
''' % (cx2_fm, cx2_fs, cx2_fk)

test1 = '''
cx2_fm = np.array([arr_(fm, fm_dtype_) for fm in iter(cx2_fm)], list)
cx2_fs = np.array([arr_(fs, fs_dtype_) for fs in iter(cx2_fs)], list)
cx2_fk = np.array([arr_(fk, fk_dtype_) for fk in iter(cx2_fk)], list)
'''
number = 1000
print('numpy comprehension: %r ' % timeit.timeit(test1, setup=setup, number=number))

test2 = '''
for cx in xrange(len(cx2_fm)):
    fm = np.array(cx2_fm[cx], dtype=FM_DTYPE)
    fm.shape = (len(fm), 2)
    cx2_fm[cx] = fm
for cx in xrange(len(cx2_fs)): 
    fs = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    cx2_fs[cx] = fs
for cx in xrange(len(cx2_fk)): 
    fk = np.array(cx2_fk[cx], dtype=FK_DTYPE)
    cx2_fk[cx] = fk
cx2_fm = np.array(cx2_fm)
cx2_fk = np.array(cx2_fk)
'''

print('iter forloop: %r' % timeit.timeit(test2, setup=setup, number=number))
