#import tpl.extern_feat.extern_feat as externf
from __future__ import division, print_function
import warnings
import os
import draw_func2 as df2
import cv2
import util
import subprocess
import numpy as np
import os, sys
from os.path import dirname, realpath, join
from PIL import Image
from numpy import uint8, float32, diag, sqrt, abs
import numpy.linalg as npla
from numpy.linalg import svd, det, inv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, Circle, FancyArrow
import tpl.extern_feat as extern_feat
import vtool.keypoint as ktool
#__file__ = 'tpl/extern_feat/extern_feat.py'
#exec(open('tpl/extern_feat/extern_feat.py').read())
#import warnings

'''
sc = 12.0158; s = 2.31244
A = np.array([( 1.79978, 0), (0.133274, 0.555623)])
array([[ 21.62579652,   0.        ], [  1.60139373,   6.67625484]])
invE = np.array([[ 0.00226126, -0.00166135], [-0.00166135,  0.0224354 ]])
'''
# Dare to compare
def draw_kpts3(kpts, method):
  with warnings.catch_warnings():
    from itertools import izip
    from matplotlib.transforms import Affine2D
    ell_linewidth=1
    ell_alpha=.5
    ell_color = (1,1,1)
    ax = plt.gca()
    kptsT = kpts.T
    (x, y, a, c, d) = kptsT
    b = np.zeros(len(a), dtype=float)
    if method == 0: # original inverse square root
        warnings.simplefilter("ignore")
        aIS = 1/np.sqrt(a)
        bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
        cIS = b
        dIS = 1/np.sqrt(d)
        #cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
    elif method == 1:
        # Just inverse
        aIS = 1/a
        bIS = -c/(a*d)
        cIS = b
        dIS = 1/d
    elif method == 2:
        # Identity
        aIS = c
        bIS = b
        cIS = c
        dIS = d
    elif method == 3:
        print('m3')
        det_ = sqrt(a*d)
        #det_ = det_**2
        print(det_)
        a/=det_
        b/=det_
        c/=det_
        d/=det_
        print(a*d)
        print(det_)
        # Modify det_
        #det_ = 1/(det_)**2
        # inverse square root
        aIS = 1/a
        bIS = -c/(a*d)
        cIS = b
        dIS = 1/d
        aIS /= sqrt(det_)
        bIS /= sqrt(det_)
        cIS /= sqrt(det_)
        dIS /= sqrt(det_)
        print(aIS*dIS)
    elif method == 4:
        print('m4')
        det_ = sqrt(a*d)
        #det_ = det_**2
        print(det_)
        a/=det_
        b/=det_
        c/=det_
        d/=det_
        print(a*d)
        print(det_)
        # Modify det_
        #det_ = 1/(det_)**2
        # inverse square root
        aIS = 1/a
        bIS = -c/(a*d)
        cIS = b
        dIS = 1/d
        aIS /= (det_)
        bIS /= (det_)
        cIS /= (det_)
        dIS /= (det_)
        print(aIS*dIS)
    elif method == 5:
        print('m5')
        # inverse square root
        A_list = ktool.get_invV_mats(kpts)
        aIS = 1/np.sqrt(a)
        bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
        cIS = b
        dIS = 1/np.sqrt(d)
    else:
        print('unknown method %r' % (method,))
        pass

    kpts_iter = izip(x,y,aIS,bIS,cIS,dIS)
    aff2d_list = [Affine2D([( a_, b_, x_),
                            ( c_, d_, y_),
                            ( 0 , 0 , 1)])
                        for (x_,y_,a_,b_,c_,d_) in kpts_iter]
    ell_actors = [Circle((0,0), 1, transform=aff2d) for aff2d in aff2d_list]
    ellipse_collection = matplotlib.collections.PatchCollection(ell_actors)
    ellipse_collection.set_facecolor('none')
    ellipse_collection.set_transform(ax.transData)
    ellipse_collection.set_alpha(ell_alpha)
    ellipse_collection.set_linewidth(ell_linewidth)
    ellipse_collection.set_edgecolor(ell_color)
    ax.add_collection(ellipse_collection)

rchip_fpath = 'tpl/extern_feat/lena.png'
rchip_fpath = 'tpl/extern_feat/zebra.jpg'
rchip_fpath = os.path.realpath(rchip_fpath)
rchip = cv2.cvtColor(cv2.imread(rchip_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
outname = extern_feat.compute_perdoch_text_feats(rchip_fpath)
#outname = compute_inria_text_feats(rchip_fpath, 'harris', 'sift')
# Keep the wrong way to compare
kpts0, desc = extern_feat.read_text_feat_file(outname)
invE = extern_feat.expand_invET(kpts0[:,2:5].T)[0]
kpts1 = extern_feat.fix_kpts_hack(kpts0[:], method=1)
A1 = ktool.get_invV_mats(kpts1)[0]
#kpts, desc = filter_kpts_scale(kpts, desc)

df2.figure(1, doclf=True)
df2.DARKEN = .5
#----
df2.imshow(rchip, plotnum=(2,3,1), title='0 before (inverse square root)')
draw_kpts3(kpts0.copy(),0)

#----
df2.imshow(rchip, plotnum=(2,3,2), title='1 just inverse')
draw_kpts3(kpts1.copy(),1)

#----
df2.imshow(rchip, plotnum=(2,3,3), title='2 identity')
draw_kpts3(kpts1.copy(), 2)
df2.update()

#----
df2.imshow(rchip, plotnum=(2,3,4), title='3 identity')
draw_kpts3(kpts1.copy(), 3)
df2.update()

#----
df2.imshow(rchip, plotnum=(2,3,5), title='inv sqrtm 4')
draw_kpts3(kpts1.copy(), 4)
df2.update()

#----
df2.imshow(rchip, plotnum=(2,3,6), title='inv sqrtm 5')
draw_kpts3(kpts1.copy(), 5)
df2.update()

# TEST
hprint = util.horiz_print
invA1 = inv(A1)
hprint('invE = ', invE)
hprint('A1 = ', A1)
hprint('invA1 = ', invA1)


exec(df2.present())
