
import cv2
from hotspotter import helpers
import textwrap
import numpy as np
import subprocess
import warnings
import numpy as np
import os, sys
from numpy import uint8, float32, diag, sqrt, abs
from numpy.linalg import det
hstr = helpers.horiz_string
hprint = helpers.horiz_print
np.set_printoptions(precision=8)

def svd(M):
    #U, S, V = np.linalg.svd(M)
    #return U,S,V
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U,S,V

def rectify_up_is_up(A):
    (a, b, c, d) = A.flatten()
    # Logic taken from Perdoch's code
    sqrt_det = np.sqrt(np.abs(a*d - b*c))
    sqrt_b2a2 = np.sqrt(b*b + a*a)
    a11 = sqrt_b2a2 / sqrt_det
    a12 = 0
    a21 = (d*b + c*a)/(sqrt_b2a2*sqrt_det)
    a22 = sqrt_det/sqrt_b2a2
    acd = np.vstack([a11, a21, a22]).T
    Aup = np.array([[a11,a12],[a21,a22]])
    return Aup, sqrt_det

def print_2x2_svd(M, name=''):
    #S, U, V = cv2.SVDecomp(M, flags=cv2.SVD_FULL_UV)
    #S = S.flatten()
    #print(hstr([U,S,V]))
    U, S, V = svd(M)
    #print(hstr([U,S,V]))
    # Try and conform to opencv
    Sm = diag(S)
    print(('---- SVD of '+name+' ----'))
    print((name+' =\n%s' % M))
    print('= U * S * V =')
    hprint([U, ' * ', Sm, ' * ', V])
    print('=')
    print((U.dot(Sm).dot(V)))
    print('-- Retified --')
    Mup, scale = rectify_up_is_up(M)
    hprint(name+'up = ', Mup)
    print(('scale=%r' % scale))



    print('--')
    return U, S, V, Sm
