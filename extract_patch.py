#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import cv2
import spatial_verification2 as sv2
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
    kp    = fx2_kp[fx:fx+1]
    subkp = np.array([(xm, ym, a, c, d)])
    return subchip, subkp

def show_feature(rchip, kp, subkp, **kwargs):
    df2.imshow(rchip, plotnum=(1,2,1), **kwargs)
    df2.draw_kpts2(kp, ell_color=(1,0,0), pts=True)
    df2.imshow(subchip, plotnum=(2,2,2), **kwargs)
    df2.draw_kpts2(subkp, ell_color=(1,0,0), pts=True)

# Get the center in the new coordinates
def border_coordinates(img):
    'specified in (x,y) coordinates'
    (img_h, img_w) = img.shape[0:2]
    tl = (0, 0)
    tr = (img_w-1, 0)
    bl = (0, img_h-1)
    br = (img_w-1, img_w-1)
    return np.array((tl, tr, bl, br)).T
def homogonize(coord_list):
    'input: list of (x,y) coordinates'
    ones_vector = np.ones((1, coord_list.shape[1]))
    coord_homog = np.vstack([np.array(coord_list), ones_vector])
    return coord_homog 
def transform_coord_list(coord_list, M):
    coord_homog  = homogonize(coord_list)
    Mcoord_homog = M.dot(coord_homog)
    Mcoord_list  = np.vstack((Mcoord_homog[0] / Mcoord_homog[2],
                              Mcoord_homog[1] / Mcoord_homog[2]))
    return Mcoord_list

def minmax_coord_list(coord_list):
    minx, miny = coord_list.min(1)
    maxx, maxy = coord_list.max(1)
    return (minx, maxx, miny, maxy)

def target_dsize(img, M):
    # Find size (and offset translation) to put new image in
    # when transforming img with M
    (img_h, img_w) = img.shape[0:2]
    coord_list   = border_coordinates(img)
    Mcoord_list = transform_coord_list(coord_list, M)
    (minx, maxx, miny, maxy) = minmax_coord_list(Mcoord_list)
    Mw, Mh = (maxx-minx, maxy-miny)
    # translate if the transformation forced any border below 0
    tx = -min(0, minx)
    ty = -min(0, miny)
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

df2.reset()

if not 'hs' in vars():
    hs = ld2.HotSpotter(params.GZ)
    cx = 111
    rchip      = hs.get_chip(cx)
    fx2_kp     = hs.feats.cx2_kpts[cx]
    fx2_desc   = hs.feats.cx2_desc[cx]
    fx2_scale  = sv2.keypoint_scale(fx2_kp)

def fx2_feature(fx):
    kp    = fx2_kp[fx:fx+1]
    desc  = fx2_desc[fx]
    scale = fx2_scale[fx]
    radius = 3*np.sqrt(3*scale)
    return kp, scale, radius, desc

def show_feature_fx(fx):
    rchip = rchip1
    kp, scale, radius, desc = fx2_feature(fx)
    subchip, subkp = get_subchip(rchip, kp, radius)
    show_feature(rchip, kp, subkp, fignum=fx)
    df2.update()


fx = 2294
kp, scale, radius, desc = fx2_feature(fx)
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

fx2_kpIS = sqrt_inv(fx2_kp)
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
def draw_sift(desc, fignum=None):
    tau = np.float64(np.pi * 2)
    NORIENTS = 8; NX = 4; NY = 4; NBINS = NX * NY
    THETA_SHIFT = tau/4
    def cirlce_rad2xy(radians, mag):
        return np.cos(radians)*mag, np.sin(radians)*mag
    discrete_theta = (np.arange(0,NORIENTS)*(tau/NORIENTS) + THETA_SHIFT)[::-1]
    # Build list of plot positions
    dim_mag   = desc / 255.0
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
    if not fignum is None:
        fig = df2.figure(fignum=fignum)
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

draw_sift(desc, fignum=10)
import sys
exec(df2.present())
#import sys
#sys.exit(1)
