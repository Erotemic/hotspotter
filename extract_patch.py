#exec(open('__init__.py').read())
from __future__ import division
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

def minmax_newradius(xtup, chip_w_, radius_):
    (x1_, x2_) = xtup
    x_ = x1_+radius_
    min_fn = lambda x:max(x,0)
    max_fn = lambda x:min(x,chip_w_)
    minmaxtup = map(min_fn, map(max_fn, (x1_, x2_)))
    x_radius_ = radius_
    if x_ - x_radius_ < 0: 
        x_radius_ = (x_radius_-x_)
    if x_ + x_radius_ > (chip_w_-1):
        x_radius_ = (chip_w-1-x_)
        x_radius_ - chip_w_ - 1 + x_ + x_radius_
    return minmaxtup, x_radius

(ix1, ix2), x_radius = minmax_newradius((x1, x2), chip_w, radius)

radius_ = radius
chip_w_ = chip_h
xtup = (y1, y2)
(iy1, iy2), y_radius = minmax_newradius((y1, y2), chip_h, radius)

minx = map(lambda x: min(x,0), (x1, x2))
(ix1, ix2) = map(lambda x: max(x,chip_w-1), minx)
miny = map(lambda y: max(y,0), (y1, y2))
(iy1, iy2) = map(lambda y: min(y,chip_h-1), miny)
#assert chip_w > ix1 > 0, 'out of bounds'
#assert chip_w > ix2 > 0, 'out of bounds'
#assert chip_h > iy1 > 0, 'out of bounds'
#assert chip_h > iy2 > 0, 'out of bounds'
x0 = x1 - chip_w + radius
y0 = y1 - chip_h + radius

H = np.array([(a, 0, x0),
              (c, d, y0),
              (0, 0,  1)])

sub_chip = rchip[y1:y2,x1:x2]



cv2.warpPerspective(rchip1,     H, rchip1.shape[0:2][::-1])


'''
this is interesting
0 - 1 = -1 
0 - 0 - 1 = -1? idk, why?
   (x - y) =    (z)
-1*(x - y) = -1*(z)
  -(x + y) =   -(z)
    -x + y = -z

let x=0
let y=1
let z=-1
   (0 - 1) =    (-1)
-1*(0 - 1) = -1*(-1)
  -(0 + 1) =   -(-1)
    -0 + 1 =    --1
    -0 + 1 = 1
         1 = 1 + 0
         1 = 1

let x=0
let a=0
let y=1
let z=-1
   (a - x - y) =    (z)
-1*(a - x - y) = -1*(z)
  -(a - x + y) =   -(z)
    -a - x + y = -z

   (0 - 0 - 1) =    (-1)
-1*(0 - 0 - 1) = -1*(-1)
  -(0 - 0 + 1) =   -(-1)
    -0 - 0 + 1 = --1

'''

