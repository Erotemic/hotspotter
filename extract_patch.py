#exec(open('__init__.py').read())
from __future__ import division
import warnings
import numpy as np
import params
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import vizualizations as viz
import helpers
import cv2
import spatial_verification2 as sv2
import sys
from numpy import sqrt

def reload_module():
    import imp, sys
    print('[extract] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def __cv2_warp_kwargs():
    flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
    borderMode = cv2.BORDER_CONSTANT
    warp_kwargs = dict(flags=flags, borderMode=borderMode)
    return warp_kwargs

#from numpy.linalg import svd
def svd(M):
    #U, S, V = np.linalg.svd(M)
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U,S,V

def draw_keypoint_patch(rchip, kp, desc=None, warped=False, **kwargs):
    #print('--------------------')
    #print('[extract] Draw Patch')
    if warped:
        wpatch, wkp  = get_warped_patch(rchip, kp)
        patch = wpatch
        subkp = wkp
    else:
        patch, subkp = get_patch(rchip, kp)
    #print('[extract] kp    = '+str(kp))
    #print('[extract] subkp = '+str(subkp))
    #print('[extract] patch.shape = %r' % (patch.shape,))
    color = (0,0,1)
    df2.imshow(patch, **kwargs)
    df2.draw_kpts2([subkp], ell_color=color, pts=True)
    if not desc is None:
        df2.draw_sift(desc, [subkp])
    #df2.draw_border(df2.plt.gca(), color, 1)

def get_warped_patch(rchip, kp):
    'Returns warped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy = kp2_sf(kp)
    sf = sfx*sfy
    #print(sf)
    s = 41#sf
    ss = sqrt(s)*3
    (h, w) = rchip.shape[0:2]
    T = np.array([[1, 0, -x],
                  [0, 1, -y],
                  [0, 0,    1]])
    sqrt_det = np.sqrt(np.sqrt(np.sqrt(a*d)))
    A = np.array([[a, 0, 0],
                  [c, d, 0],
                  [0, 0, 1]])
    A = np.linalg.inv(A)
    S2 = np.array([[ss,  0,  0],
                   [0,  ss,  0],
                   [0 ,  0,  1]])
    X = np.array([[1, 0, s/2],
                  [0, 1, s/2],
                  [0, 0,   1]])
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    inv = np.linalg.inv
    det = np.linalg.det
    M = X.dot(S2).dot(A).dot(T)
    warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **__cv2_warp_kwargs())
    #warped_patch = cv2.warpPerspective(rchip, M, dsize, **__cv2_warp_kwargs())
    wkp = np.array([(s/2, s/2, ss, 0., ss)])
    return warped_patch, wkp
    

def get_patch(rchip, kp):
    'Returns cropped unwarped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy =  kp2_sf(kp)
    ratio = max(sfx, sfy) / min(sfx, sfy)
    radx = sfx * ratio
    rady = sfy * ratio
    #print('[get_patch] sfy=%r' % sfy)
    #print('[get_patch] sfx=%r' % sfx)
    #print('[get_patch] ratio=%r' % ratio)

    (chip_h, chip_w) = rchip.shape[0:2]
    ix1, ix2, xm = quantize_to_pixel_with_offset(x, radx, 0, chip_w)
    iy1, iy2, ym = quantize_to_pixel_with_offset(y, rady, 0, chip_h)
    patch = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy() # subkeypoint in patch coordinates
    subkp[0:2] = (xm, ym)
    return patch, subkp


def quantize_to_pixel_with_offset(z, radius, low, high):
    ''' Quantizes a small area into an indexable pixel location 
    Returns: pixel_range=(iz1, iz2), subpxl_offset
    Pixels:
    +   ___+___+___          +
    ^        ^ ^             ^
    z1       z iz           z2              
            _______________ < radius
                _____________ < quantized radius '''      
    (z1, z2) = (z-radius, z+radius)
    iz1 = max(np.floor(z1), low)
    iz2 = min(np.ceil(z2), high)
    z_radius1 = np.ceil(z - iz1)
    z_radius2 = np.ceil(iz2 - z)
    z_radius = min(z_radius1, z_radius2)
    (iz1, iz2) = (z-z_radius, z+z_radius)
    z_radius = np.ceil(z - iz1)
    return iz1, iz2, z_radius

def kp2_sf(kp):
    'computes scale factor of keypoint'
    (x, y, a, c, d) = kp
    A = np.array(((a,0),(c,d)))
    U,S,V = svd(A)
    #sf = np.sqrt(1/(a*d))
    sfx = S[1]
    sfy = S[0]
    return sfx, sfy


