#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import cv2
import spatial_verification2 as sv2
import sys
import params

def __cv2_warp_kwargs():
    flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
    borderMode = cv2.BORDER_CONSTANT
    warp_kwargs = dict(flags=flags, borderMode=borderMode)
    return warp_kwargs

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

def get_subchip(rchip, kp, radius):
    x, y, a, c, d = kp[0]
    (chip_h, chip_w) = rchip.shape[0:2]
    ix1, ix2, xm = quantize_to_pixel_with_offset(x, radius, 0, chip_w)
    iy1, iy2, ym = quantize_to_pixel_with_offset(y, radius, 0, chip_h)
    subchip = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy() # subkeypoint in subchip coordinates
    subkp[0,0:1] = (xm, ym)
    return subchip, subkp

def show_feature(rchip, kp, subkp, **kwargs):
    df2.imshow(rchip, plotnum=(1,2,1), **kwargs)
    df2.draw_kpts2(kp, ell_color=(1,0,0), pts=True)
    df2.imshow(subchip, plotnum=(2,2,2), **kwargs)
    df2.draw_kpts2(subkp, ell_color=(1,0,0), pts=True)

def target_dsize(img, M):
    # Get img bounds under transformation
    (minx, maxx, miny, maxy) = sv2.transformed_bounds(img, M)
    Mw, Mh = (maxx-minx, maxy-miny)
    # If any border forced below, return a translation to append to M
    tx = -min(0, minx)
    ty = -min(0, miny)
    # Round to integer size
    dsize = tuple(map(int, np.ceil((Mw, Mh))))
    return dsize, tx, ty

def warp_image(img, M):
    img_size = img.shape[0:2]
    # Find the target warped img extent, add any tranlations
    dsize, tx, ty = target_dsize(img, M)
    M = M.copy()
    M[0,2] += tx
    M[1,2] += ty
    print('warp %r -> %r' % (img_size, dsize))
    #warp_img = cv2.warpAffine(img, M[0:2], dsize, **__cv2_warp_kwargs())
    warp_img = cv2.warpPerspective(img, M, dsize, **__cv2_warp_kwargs())
    return warp_img, tx, ty, M

df2.reset()

if not 'hs' in vars():
    hs = ld2.HotSpotter()
    hs.load_all(params.GZ)
    qcx = 111
    cx = 305
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    (fx2_kp, fx2_desc, fx2_scale) = hs.get_features(qcx)
    fx2_feature = hs.get_feature_fn(cx)
    df2.show_matches(rchip1, rchip2, kpts1, kpts2, fm)

def show_feature_fx(fx):
    rchip = rchip1
    kp, scale, radius, desc = fx2_feature(fx)
    subchip, subkp = get_subchip(rchip, kp, radius)
    show_feature(rchip, kp, subkp, fignum=fx)
    df2.update()

df2.show_matches(

fx = 2294
(kp, scale, radius, desc) = fx2_feature(fx)
subchip, subkp = get_subchip(rchip, kp, radius)

show_feature(rchip, kp, subkp, fignum=fx)

(x, y, a, c, d) = kp[0]
# Transformation from ellipse to a unit circle
A = np.array([(a, 0, 0),
              (c, d, 0),
              (0, 0, 1)])
# Scale up so we can see the keypoint
sf = 1000
S = np.array([(sf*3,  0, 0), 
              ( 0, sf*3, 0),
              ( 0,  0, 1)])

print('A')
print A
A = A.dot(S)
print A

fx2_kpIS = sv2.sqrt_inv(fx2_kp)
kpIS = fx2_kpIS[fx]
print('warp subchip')
#subchip2 = np.swapaxes(subchip, 0, 1)
warp_subchip, tx, ty, M = warp_image(subchip, A)
#warp_subchip = np.swapaxes(warp_subchip, 0, 1)

print('warp_subchip.shape = %r ' % (warp_subchip.shape,))
#circle_a = 3/(np.sqrt(3*sf))
xm = subkp[0,0]
ym = subkp[0,1]
[Axm, Aym,_]  = M.dot(np.array([[xm],[ym],[1]])).flatten()

circle_a = 1/sf
#(sf*3*np.sqrt(3))
unit_circle = np.array([(Axm, Aym, circle_a, 0., circle_a)])

# Plot full, cropped, warp


df2.imshow(warp_subchip, plotnum=(2,2,4), fignum=fx)
df2.draw_kpts2(unit_circle, ell_color=(1,0,0), pts=True)

#----

fig = df2.figure(fignum=10)
ax = df2.plt.gca()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
#---
df2.draw_sift(desc)

exec(df2.present())
#import sys
#sys.exit(1)
