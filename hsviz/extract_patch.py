#exec(open('__init__.py').read())
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[extract]', DEBUG=False)
# Science
import numpy as np
# Hotspotter
import draw_func2 as df2
from hscom import util
import vtool.patch as ptool


def draw_warped_keypoint_patch(rchip, kp, **kwargs):
    return draw_keypoint_patch(rchip, kp, warped=True, **kwargs)


@util.indent_decor('[extract_patch.dkp]')
def draw_keypoint_patch(rchip, kp, sift=None, warped=False, **kwargs):
    #print('--------------------')
    #print('[extract] Draw Patch')
    kpts = np.array([kp])
    if warped:
        wpatches, wkpts = ptool.get_warped_patches(rchip, kpts)
        patches = wpatches
        subkpts = wkpts
    else:
        patches, subkpts = ptool.get_unwarped_patches(rchip, kpts)
    #print('[extract] kpts[0]    = %r' % (kpts[0]))
    #print('[extract] subkpts[0] = %r' % (subkpts[0]))
    #print('[extract] patches[0].shape = %r' % (patches[0].shape,))
    color = (0, 0, 1)
    patch = patches[0]
    fig, ax = df2.imshow(patch, **kwargs)
    sifts = np.array([sift])
    subkpts_ = np.array(subkpts)
    df2.draw_kpts2(subkpts_, ell_color=color, pts=True, sifts=sifts)
    #if not sift is None:
        #df2.draw_sift(sift, subkpts_[0])
    return ax
    #df2.draw_border(df2.gca(), color, 1)
