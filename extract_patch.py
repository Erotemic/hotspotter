#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
if not 'hs' in vars():
    (hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2) = ld2.get_sv_test_data()

import draw_func2 as df2
import cv2

df2.imshow(rchip1)
import spatial_verification2 as sv2
def keypoint_scale(fx2_kp):
    fx2_acd = fx2_kp[:,2:5].T
    fx2_det = sv2.det_acd(fx2_acd)
    fx2_scale = np.sqrt(1/fx2_det)
    return fx2_scale

fx2_kp     = kpts1
fx2_scale  = keypoint_scale(fx2_kp)
fx2_radius = 2 * 3*np.sqrt(3*fx2_scale)

rchip = rchip1
(x,y,a,c,d) = fx2_kp[0]
scale = fx2_scale[0]
radius = fx2_radius[0]
chip_w, chip_h = rchip.shape[0:2]
(x1, x2) = (x-radius, x+radius)
(y1, y2) = (y-radius, y+radius)
print('x=%r, y=%r' % (x,y))
print('radius = %r' % radius)
print('rchip.shape = %r' % (rchip.shape,))

def minmax(x1,x2, low, high):
    min_fn = lambda x:max(x,low)
    max_fn = lambda x:min(x,high)
    (ix1,ix2) = map(int, map(min_fn, map(max_fn, map(round, (x1, x2)))))
    return (ix1, ix2)

print('------')

print('lets think about x')
print('x = %r' % x)
print('radius = %r' % radius)
(x1, x2) = (x-radius, x+radius)
(ix1, ix2) = minmax(x1, x2, 0, chip_w)
print('ideal x1:x2 = %r:%r' % (x1, x2))
print('truncated ix1:ix2 = %r:%r' % (ix1, ix2))
rounderr1 = x1 - ix1
rounderr2 = x2 - ix2

print x - (ix1 + radius), x - (ix2 - radius)
x_radius = radius + min(rounderr1, rounderr2)
print x_radius 
print('round_err = %r,%r' % (rounderr1, rounderr2))
import sys
sys.exit(1)
print('------')
def minmax_newradius(x_, radius_, chip_w_):
    '''Used to ensure good cutout coordinates and 
    fixes the radius if anything has been truncated'''
    (x1_, x2_) = (x_-radius_, x_+radius_)
    print('range_ = %5.2f:%5.2f' % (x1_, x2_))
    x_ = x1_+radius_
    rnd_fn = lambda x:round(x)
    print('round_diff')
    minmaxtup = map(int, map(min_fn, map(max_fn, map(round, (x1_, x2_)))))
    #print(x1_ - minmaxtup[0], x2_ - minmaxtup[1])
    x_radius_ = radius_ - minmaxtup[0] + x1_
    if x_ - x_radius_ < 0: 
        x_radius_ = x_
    if x_ + x_radius_ > (chip_w_-1):
        x_radius_ = (chip_w-1-x_)
    return minmaxtup, x_radius_

(ix1, ix2), x_radius = minmax_newradius(x, radius, chip_w)
(iy1, iy2), y_radius = minmax_newradius(y, radius, chip_h)
print('ix_range = [%3d:%3d], x_radius = %r' % (ix1, ix2, x_radius))
print('check = %r' % ((ix1+x_radius)-(x-radius)))

print('iy_range = [%3d:%3d], y_radius = %r' % (iy1, iy2, y_radius))
print('check = %r, %r' % (iy1+x_radius, y-radius))

radius_ = radius
chip_w_ = chip_h
xtup = (y1, y2)

x0 = x1 - chip_w + radius
y0 = y1 - chip_h + radius

H = np.array([(a, 0, x0),
              (c, d, y0),
              (0, 0,  1)])

#print('subchip.shape = %r', (subchip.shape,))

#sub_chip = rchip[y1:y2,x1:x2]

#cv2.warpPerspective(rchip1,     H, rchip1.shape[0:2][::-1])
