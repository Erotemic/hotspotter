#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import cv2
import spatial_verification2 as sv2

df2.reset()

if not 'hs' in vars():
    (hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2) = ld2.get_sv_test_data()

def __warp_kwargs():
    # Set cv2 flags
    flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
    borderMode = cv2.BORDER_CONSTANT
    warp_kwargs = dict(flags=flags, borderMode=borderMode)
    return warp_kwargs

def warp_border(img, M):
    #print('------------')
    #print('M = \n%r ' % (M,))
    #print('img.shape=%r ' % (img.shape,))
    #print('img.max()=%r, img.min()=%r ' % (img.max(), img.min()))
    #print('img.dtype=%r ' % (img.dtype,))
    coord_list   = np.array(border_coordinates(img))
    coord_homog  = homogonize(coord_list)
    Mcoord_homog = M.dot(coord_homog)
    Mcoord_list = np.vstack((Mcoord_homog[0] / Mcoord_homog[-1],
                             Mcoord_homog[1] / Mcoord_homog[-1])).T
    #print('coord_list: ')
    #print(coord_list.T)
    #print('Mcoord_list: ')
    #print(Mcoord_list.T)
    Mxywh = border2_xywh(Mcoord_list)
    #print('Mxywh')
    #print(Mxywh)
    #print('------------')

#print('Rchip warp test')
#print('subchip warp test')
#warp_border(subchip, A)
#print target_scale_factor(rchip1, H)
#print target_scale_factor(subchip, A)

fx2_kp     = kpts1
fx2_desc   = hs.feats.cx2_desc[cx]
fx2_scale  = sv2.keypoint_scale(fx2_kp)
fx2_radius = np.sqrt(3*fx2_scale)

SEL = 2000

rchip = rchip1
(x,y,a,c,d) = fx2_kp[SEL]
desc = fx2_desc[SEL]
scale = fx2_scale[SEL]
radius = fx2_radius[SEL]
(chip_h, chip_w) = rchip.shape[0:2]

def minmax(z1, z2, low, high):
    iz1 = max(np.floor(z1), low)
    iz2 = min(np.ceil(z2), high)
    return (iz1, iz2)

def int_window_and_midpoint(z, radius, low, high):
    (z1, z2) = (z-radius, z+radius)
    print(z1)
    print(z2)
    print(high)
    (iz1, iz2) = minmax(z1, z2, low, high)
    radius_z1 = z - iz1
    zm = radius_z1
    return iz1, iz2, zm

print('-----')
print('rchip.shape = %r' % (rchip.shape,))
print('(y,x)  = (%.2f, %.2f)' % (y,x))
print('scale  = %.4f' % scale)
print('radius = %.4f' % radius)

ix1, ix2, xm = int_window_and_midpoint(x, radius, 0, chip_w)
iy1, iy2, ym = int_window_and_midpoint(y, radius, 0, chip_h)
subchip = rchip[iy1:iy2,ix1:ix2]

print('-----')
print('subchip = rchip[%d:%d, %d:%d]' % (iy1, iy2, ix1, ix2))
print('subchip.shape = %r' % (map(int,subchip.shape),))
print('-----')

subkp = np.array([(xm, ym, a, c, d)])

# Transformation from ellipse to a unit circle
A = np.array([(a, 0, 0),
              (c, d, 0),
              (0, 0, 1)])
# Scale up so we can see the keypoint
sf = 100
S = np.array([(sf**2,  0, 0), 
              ( 0, sf**2, 0),
              ( 0,  0, 1)])
print('A')
print A
A = A.dot(S)
print A

# Get the center in the new coordinates

def border_coordinates(img):
    'specified in (x,y) coordinates'
    (img_h, img_w) = img.shape[0:2]
    tl = (0, 0)
    tr = (img_w-1, 0)
    bl = (0, img_h-1)
    br = (img_w-1, img_w-1)
    return (tl, tr, bl, br)

def border2_xywh(coord_list):
    (tl, tr, bl, br) = coord_list
    (x,y) = tl
    (w,h) = np.array(br) - np.array(tl)
    xywh = (x,y,w,h)
    return xywh

def homogonize(coord_list):
    'input: list of (x,y) coordinates'
    coord_homog = np.hstack([np.array(coord_list), np.ones((len(coord_list),1))]).T
    return coord_homog 

def target_dsize(img, M):
    # Given an image. Transformation M will warp it
    (img_h, img_w) = img.shape[0:2]
    coord_list   = np.array(border_coordinates(img))
    coord_homog  = homogonize(coord_list)
    print coord_homog
    Mcoord_homog = M.dot(coord_homog)
    # The borders will be in this position
    Mcoord_list = np.vstack((Mcoord_homog[0] / Mcoord_homog[-1],
                             Mcoord_homog[1] / Mcoord_homog[-1])).T
    print Mcoord_homog
    # so the range of the transformation would yeild:
    Mxywh = border2_xywh(Mcoord_list)

    minx = Mcoord_homog[0].min()
    maxx = Mcoord_homog[0].max()
    miny = Mcoord_homog[1].min()
    maxy = Mcoord_homog[1].max()
    # but lets fit this in better coordinates
    Mw, Mh = (maxx-minx, maxy-miny)
    tx = -min(0,minx)
    ty = -min(0,miny)
    print('target size: w=%d, h=%d' % (Mw, Mh))
    print('target tx=%r, ty=%r' % (tx, ty))
    dsize = tuple(map(int, np.ceil((Mw, Mh))))
    return dsize, tx, ty

def warp_image(img, M):
    img_size = img.shape[0:2]
    dsize, tx, ty = target_dsize(img, M)
    M = M.copy()
    M[0,2] += tx
    M[1,2] += ty
    print('warp %r -> %r' % (img_size, dsize))
    warp_img = cv2.warpAffine(img, M[0:2], dsize, **__warp_kwargs())
    return warp_img, tx, ty, M

def sqrt_inv(kpts):
    x, y, a, c, d = kpts.T
    aIS = 1/np.sqrt(a) 
    bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
    dIS = 1/np.sqrt(d)
    kpts_iter = iter(zip(x,y,aIS,bIS,dIS))
    kptsIS = [np.array([( a_, b_, x_),
                        ( 0 , d_, y_),
                        ( 0 , 0 , 1)])
              for (x_,y_,a_,b_,d_) in kpts_iter ]
    return kptsIS
    

fx2_kpIS = sqrt_inv(fx2_kp)
kpIS = fx2_kpIS[SEL]
print('warp subchip')
#subchip2 = np.swapaxes(subchip, 0, 1)
warp_subchip, tx, ty, M = warp_image(subchip, A)
#warp_subchip = np.swapaxes(warp_subchip, 0, 1)

print('warp_subchip.shape = %r ' % (warp_subchip.shape,))
#circle_a = 3/(np.sqrt(3*sf))
[Axm, Aym,_]  = M.dot(np.array([[xm],[ym],[1]])).flatten()

circle_a = 1/sf
#(sf*3*np.sqrt(3))
unit_circle = np.array([(Axm, Aym, circle_a, 0., circle_a)])

# Plot full, cropped, warp

df2.imshow(rchip, plotnum=(3,1,1), fignum=SEL)
df2.draw_kpts2(fx2_kp[SEL:SEL+1], ell_color=(1,0,0), pts=True)

df2.imshow(subchip, plotnum=(3,1,2), fignum=SEL)
df2.draw_kpts2(subkp, ell_color=(1,0,0), pts=True)

df2.imshow(warp_subchip, plotnum=(3,1,3), fignum=SEL)
df2.draw_kpts2(unit_circle, ell_color=(1,0,0), pts=True)

df2.present(num_rc=(2,1),wh=(800,1000))


#----
def draw_sift(desc):
    NORIENTS = 8
    NX = 4
    NY = 4
    NBINS = NX * NY
    tau = np.float64(np.pi * 2)
    def cirlce_rad2xy(radians, mag):
        return np.cos(radians)*mag, np.sin(radians)*mag

    THETA_SHIFT = tau/4
    discrete_theta = (np.arange(0,NORIENTS)*(tau/NORIENTS) + THETA_SHIFT)[::-1]


    dim_mag   = desc / 256.0
    dim_theta = np.tile(discrete_theta, (NBINS, 1)).flatten()
    dim_xy = np.array(zip(*cirlce_rad2xy(dim_theta, dim_mag))) 
    def xyt_gen(): 
        for x in xrange(NX): 
            for y in xrange(NY):
                for t in xrange(NORIENTS):
                    yield x,y,t
    def xy_gen():
        for x in xrange(NX):
            for y in xrange(NY):
                    yield x,y
    
    fig = df2.figure(fignum=40)
    ax = df2.plt.gca()
    ax.set_xlim(-6,36)
    ax.set_ylim(-6,36)
    ax.set_aspect('equal')

    DSCALE = 5
    XYSCALE = 10

    # Draw Arms
    for x,y,t in xyt_gen():
        index = x*(NY*NORIENTS)+y*(NORIENTS) + t
        (dx, dy) = dim_xy[index]
        x_data = [x*XYSCALE, x*XYSCALE + dx*DSCALE]
        y_data = [y*XYSCALE, y*XYSCALE + dy*DSCALE]
        dim_artist = df2.plt.Line2D(x_data, y_data, color=(0,0,1))
        ax.add_artist(dim_artist)
    # Draw Circles
    for x,y in xy_gen():
        circ_artist = df2.plt.Circle((x*XYSCALE, y*XYSCALE), DSCALE, color=(1,0,0))
        circ_artist.set_facecolor('none')
        ax.add_artist(circ_artist)

draw_sift(desc)
df2.present()
#import sys
#sys.exit(1)
