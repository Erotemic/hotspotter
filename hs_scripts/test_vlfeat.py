import vlfeat as vl
import cv2
import numpy as np

'''
vl_quickvis(I, ratio, kernelsize, maxdist, maxcuts=None)
    Create an edge image from a Quickshift segmentation.
    IEDGE = VL_QUICKVIS(I, RATIO, KERNELSIZE, MAXDIST, MAXCUTS) creates an edge
    stability image from a Quickshift segmentation. RATIO controls the tradeoff
    between color consistency and spatial consistency (See VL_QUICKSEG) and
    KERNELSIZE controls the bandwidth of the density estimator (See VL_QUICKSEG,
    VL_QUICKSHIFT). MAXDIST is the maximum distance between neighbors which
    increase the density. 
'''

I = cv2.imread('/lena.png')
ratio = .5
kernelsize=10
maxdist=10.0

vl.vl_quickvis(I, ratio, kernelsize, maxdist)
