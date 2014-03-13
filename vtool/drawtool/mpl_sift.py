from __future__ import division, print_function
# Standard
from itertools import product as iprod
from itertools import izip
# Science
import numpy as np
# Matplotlib
import matplotlib as mpl
np.tau = 2 * np.pi  # tauday.com


BLACK  = np.array((0.0, 0.0, 0.0, 1.0))
RED    = np.array((1.0, 0.0, 0.0, 1.0))


def _cirlce_rad2xy(radians, mag):
    return np.cos(radians) * mag, np.sin(radians) * mag


def _set_colltup_list_transform(colltup_list, trans):
    for coll_tup in colltup_list:
        for coll in coll_tup:
            coll.set_transform(trans)


def _draw_colltup_list(ax, colltup_list):
    for coll_tup in colltup_list:
        for coll in coll_tup:
            ax.add_collection(coll)


# Create a patch collection with attributes
def _circl_collection(patch_list, color, alpha):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_alpha(alpha)
    coll.set_edgecolor(color)
    coll.set_facecolor('none')
    return coll


def _arm_collection(patch_list, color, alpha, lw):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_alpha(alpha)
    coll.set_color(color)
    coll.set_linewidth(lw)
    return coll


def get_sift_collection(sift,
                        aff=mpl.transforms.Affine2D(),
                        bin_color=BLACK,
                        arm1_color=RED,
                        arm2_color=BLACK):
    # global offset scale adjustments
    DSCALE   =  0.25
    ARMSCALE =  1.5
    XYSCALE  =  0.5
    XYOFFST  = -0.75
    # SIFT CONSTANTS
    NORIENTS, NX, NY = 8, 4, 4
    NBINS = NX * NY
    # Build list of plot positions
    # Build an "arm" for each sift measurement
    discrete_ori = (np.arange(0, NORIENTS) * (np.tau / NORIENTS))
    arm_mag   = sift / 255.0
    arm_ori = np.tile(discrete_ori, (NBINS, 1)).flatten()
    # The offset x,y's for each sift measurment
    arm_dxy = np.array(zip(*_cirlce_rad2xy(arm_ori, arm_mag)))
    yxt_gen = iprod(xrange(NY), xrange(NX), xrange(NORIENTS))
    yx_gen  = iprod(xrange(NY), xrange(NX))
    # Draw 8 directional arms in each of the 4x4 grid cells
    arrow_patches1 = []
    arrow_patches2 = []
    _kwarm = dict(head_width=1e-10, length_includes_head=False, transform=aff)
    _kwcirc = dict(transform=aff)
    for y, x, t in yxt_gen:
        index = y * NX * NORIENTS + x * NORIENTS + t
        (dx, dy) = arm_dxy[index]
        arm_x  = x * XYSCALE + XYOFFST
        arm_y  = y * XYSCALE + XYOFFST
        arm_dy = dy * DSCALE * ARMSCALE
        arm_dx = dx * DSCALE * ARMSCALE
        _args = [arm_x, arm_y, arm_dx, arm_dy]
        arrow_patches1.append(mpl.patches.FancyArrow(*_args, **_kwarm))
        arrow_patches2.append(mpl.patches.FancyArrow(*_args, **_kwarm))
    # Draw circles around each of the 4x4 grid cells
    circle_patches = []
    for y, x in yx_gen:
        circ_xy = (x * XYSCALE + XYOFFST, y * XYSCALE + XYOFFST)
        circ_radius = DSCALE
        circle_patches += [mpl.patches.Circle(circ_xy, circ_radius, **_kwcirc)]

    circ_coll = _circl_collection(circle_patches,  bin_color, 0.5)
    arm1_coll = _arm_collection(arrow_patches1, arm1_color, 1.0, 0.5)
    arm2_coll = _arm_collection(arrow_patches2, arm2_color, 1.0, 1.0)
    coll_tup = (circ_coll, arm2_coll, arm1_coll)
    return coll_tup


def draw_sifts(ax, sifts, invVR_aff2Ds=None, **kwargs):
    if invVR_aff2Ds is None:
        invVR_aff2Ds = [mpl.transforms.Affine2D() for _ in xrange(len(sifts))]
    colltup_list = [get_sift_collection(sift, aff, **kwargs) for sift, aff in izip(sifts, invVR_aff2Ds)]
    _set_colltup_list_transform(colltup_list, ax.transData)
    _draw_colltup_list(ax, colltup_list)
