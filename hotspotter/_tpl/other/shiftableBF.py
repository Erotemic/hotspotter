# Adapted from Matlab to Python by Jon Crall
# Original Matlab Source: 
# http://www.mathworks.se/matlabcentral/fileexchange/36657-fast-bilateral-filter/content/FastBilateralFilter/shiftableBF.m
# These are shorthands I used to help with porting 1 based to 0 based
# [k]    =        2:end  
# [-k-1] = fliplr(1:end-1) 
# [-k]   = fliplr(2:end)

''' Filtering operation

function shiftableBF:
    inImg      :  grayscale image
    sigmaS     : width of spatial Gaussian
    sigmaR     : width of range Gaussian
    [-w, w]^2  : domain of spatial Gaussian
    tol        : truncation error  

Author: Kunal N. Chaudhury
Date:   March 1, 2012

Converted to python on March 25 2013 by Jon Crall

Reference: 
[1] K.N. Chaudhury, D. Sage, and M. Unser, 'Fast O(1) bilateral  filtering using
trigonometric range kernels,' IEEE Transactions on Image Processing, vol. 20,
no. 11, 2011.
[2] K.N. Chaudhury, 'Acceleration of the shiftable O(1) algorithm for bilateral filtering
and non-local means,' arXiv:1203.5128v1.  '''

from pylab import find
from scipy.misc import comb
from scipy.ndimage import convolve
from scipy.signal import gaussian
from skimage.color import rgb2gray
import numpy as np
import sys

def maxFilter(inImg, w):
    ''' Computes the maximum 'local' dynamic range
    inImg       : grayscale image
    [-w, w]^2   : search window (w must be odd)
    T           : maximum local dynamic range '''
    T = -1
    sym    = (w - 1.)/2.
    (m, n) = inImg.shape
    pad1 =  int(w*np.ceil(float(m)/float(w)) - m)
    pad2 =  int(w*np.ceil(float(n)/float(w)) - n)
    inImg2 = np.pad(inImg, ((0, pad1), (0,pad2)), mode='symmetric')
    template = inImg2.copy()
    m = m + pad1
    n = n + pad2
    # scan along row
    for ii in xrange(0,m):
        L      = np.zeros(n) # From the Left
        R      = np.zeros(n) # From the Right
        L[0]   = template[ii, 0]
        R[n-1] = template[ii, n-1]
        for k in xrange(1, n):
            if k % w == 0:
                L[k]    = template[ii,  k  ]
                R[-k-1] = template[ii, -k-1]
            else:
                L[k]     = max(L[k-1], template[ii,    k])
                R[-k-1]  = max(R[ -k], template[ii, -k-1])
        for k in xrange(0, n):
            p = k-sym; q = k+sym
            template[ii, k] = max\
                    ( R[p] if p >= 0 else -1,
                      L[q] if q <  n else -1)
    # scan along column
    for jj in xrange(0, n):
        L      = np.zeros(m)
        R      = np.zeros(m)
        L[0]   = template[0,   jj]
        R[m-1] = template[m-1, jj]
        for k in xrange(1, m):
            if k % w == 0:
                L[k]      = template[k,   jj] 
                R[-k-1]  = template[-k-1, jj] 
            else:
                L[k]     = max(L[k-1], template[ k  , jj])
                R[-k-1]  = max(R[ -k], template[-k-1, jj])
        for k in xrange(0, m):
            p = k-sym; q = k+sym
            temp = max\
                    (R[p] if p >= 0 else -1,
                     L[q] if q <  m else -1) - inImg2[k, jj]
            if temp > T:
                T = temp
    return T

# These take a long time: (chip, 1.6, 10, 7, 0)
# These are almost as good: (chip, 1.6, 200, 7, 0)
# SigmaR is the real time sucker. The bigger it is the faster it goes
def shiftableBF(inImg, sigmaS=1.6, sigmaR=200, w=7, tol=0):
    '''
    inImg - expects a grayscale numpy array with dtype=uint8
    np.asarray(Image.open(imname).convert('L'))
    '''
    inMax = inImg.max
    nChan = 1 if len(inImg.shape) == 2 else inImg.shape[2]
    inTyp = inImg.dtype
    if nChan == 4: inImg = inImg[:,:,0:3]  # remove alpha
    if nChan  > 1: inImg = rgb2gray(inImg) # remove color
    if inMax <= 1: inImg *= 255.           # force to range 0,255

    if w % 2 == 0: 
        w = w + 1
    g = gaussian(w,sigmaS)
    g = g.reshape((w,1)) / sum(g)
    filt = g*np.transpose(g)

    # set range interval and the order of raised cosine
    #T  =  maxFilter(inImg, w)
    T = 192
    #print 'T= %r' % T
    N  =  float(np.ceil( 0.405 * (float(T) / float(sigmaR))**2 ))
    #print 'N= %r' % N
    gamma    =  1. / (np.sqrt(N) * sigmaR)
    #print 'gamma= %r' % gamma
    twoN     =  2**N
    #print 'N^2= %r' % twoN

    # compute truncation
    if tol == 0:
        M = 0
    else:
        if sigmaR > 40:
            #print "SigmaR > 4, setting M to 0"
            M = 0
        elif sigmaR > 10:
            #print "SigmaR > 10, adjusting to tolerence"
            sumCoeffs = 0
            M = -1
            #print "Trying N choose from from 0 to %d " % int(np.round(N/2)+1)
            for k in xrange(0,int(np.round(N/2)+1)):
                sumCoeffs += comb(N,k)/twoN
                if sumCoeffs > tol/2.:
                    #print "sumCoeeffs %d  tol/2 = %f" % (sumCoeffs, tol/2)
                    M = k
                    break
            if M == -1:
                #print "Setting to 0"
                M = 0
        else:
            #print "40 > SigmaR > 10, adjusting to tolerence"
            M = np.ceil( 0.5 * ( N - np.sqrt(4 * N * np.log10(2./tol)) ) )
        
    #print 'M = %r' % M
    # main filter
    (m, n)   =  inImg.shape
    outImg1  =  np.zeros((m, n))
    outImg2  =  np.zeros((m, n))
    outImg   =  np.zeros((m, n))

    #print (M, N-M+1)
    #sys.stdout.flush()

    for k in np.arange(M, N-M+1):
        coeff = comb(N,k) / twoN
        
        cosImg  = np.cos( (2.*k-N) * gamma * inImg )
        sinImg  = np.sin( (2.*k-N) * gamma * inImg )
         
        
        phi1 = convolve(np.multiply(inImg, cosImg), filt)
        phi2 = convolve(np.multiply(inImg, sinImg), filt)
        phi3 = convolve(cosImg, filt)
        phi4 = convolve(sinImg, filt)

        outImg1 += coeff * np.add(np.multiply(cosImg, phi1), np.multiply(sinImg, phi2))
        outImg2 += coeff * np.add(np.multiply(cosImg, phi3), np.multiply(sinImg, phi4))
        
    # avoid division by zero

    inImg.shape   = inImg.size
    outImg.shape  = outImg.size
    outImg1.shape = outImg1.size
    outImg2.shape = outImg2.size 

    idx1 = find( outImg2 < 0.0001)
    idx2 = find( outImg2 > 0.0001)

    outImg[ idx1 ] = inImg[idx1]
    outImg[ idx2 ] = np.divide(outImg1[idx2], outImg2[idx2])

    inImg.shape   = (m,n)
    outImg.shape  = (m,n)
    outImg1.shape = (m,n)
    outImg2.shape = (m,n)

    # keep output consistent with input
    if outImg.max() <= 1 and inMax > 1:
        outImg *= 255.
    elif inMax <= 1 and outImg.max() > 1:
        outImg /= 255.
    if outImg.dtype != inTyp:
        outImg = np.array(outImg,dtype=inTyp)

    return outImg

