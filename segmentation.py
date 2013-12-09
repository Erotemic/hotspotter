#from __init__ import *
from __future__ import division
import numpy as np
import helpers
import cv2
import algos
import draw_func2 as df2
import sys

DEBUG_SEGM = False

def reload_module():
    import imp, sys
    print('[seg] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def printDBG(msg):
    if DEBUG_SEGM:
        print(msg)
    pass

def im(img, fignum=0):
    df2.imshow(img, fignum=fignum)
    df2.update()

def resize_img_and_roi(img_fpath, roi_, new_size=None, sqrt_area=400.0):
    printDBG('[segm] imread(%r) ' % img_fpath)
    full_img = cv2.cvtColor(cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    (full_h, full_w) = full_img.shape[:2]                 # Image Shape
    printDBG('[segm] full_img.shape=%r' % (full_img.shape,))
    (rw_, rh_) = roi_[2:]
    # Ensure that we know the new chip size
    if new_size is None:
        target_area = float(sqrt_area) ** 2
        def _resz(w, h):
            ht = np.sqrt(target_area * h / w)
            wt = w * ht / h
            return (int(round(wt)), int(round(ht)))
        new_size_ = _resz(rw_, rh_)
    else:
        new_size_ = new_size 
    # Get Scale Factors
    fx = new_size_[0] / rw_
    fy = new_size_[1] / rh_
    printDBG('[segm] fx=%r fy=%r' % (fx, fy))
    dsize = (int(round(fx*full_w)), int(round(fy*full_h)))
    printDBG('[segm] dsize=%r' % (dsize,))
    # Resize the image
    img_resz = cv2.resize(full_img, dsize, interpolation=cv2.INTER_LANCZOS4)
    # Get new ROI in resized image
    roi_resz = np.array(np.round(roi_ * fx), dtype=np.int64)
    return img_resz, roi_resz

def test(hs, cx=0):
    import draw_func2 as df2
    import os
    if not 'cx' in vars():
        cx = 0
    # READ IMAGE AND ROI
    cx2_roi = hs.tables.cx2_roi
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname
    #---
    roi_ = cx2_roi[cx]
    gx  = cx2_gx[cx]
    img_fname = gx2_gname[gx]
    img_fpath = os.path.join(hs.dirs.img_dir, img_fname)
    #---
    print('testing segment')
    seg_chip, img_mask = segment(img_fpath, roi_, new_size=None)
    df2.show_img(hs, cx, fignum=1, plotnum=131, title='original', doclf=True)
    df2.imshow(img_mask, fignum=1, plotnum=132, title='mask')
    df2.imshow(seg_chip, fignum=1, plotnum=133, title='segmented')

def clean_mask(mask, num_dilate=3, num_erode=3, window_frac=.025):
    '''Clean the mask
    (num_erode, num_dilate) = (1, 1)
    (w, h) = (10, 10)'''
    w = h = int(round(min(mask.shape) * window_frac))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(w,h))
    _mask = mask
    # compute the closing
    for ix in xrange(num_dilate):
        _mask = cv2.dilate(_mask, element)
    for ix in xrange(num_erode):
        _mask = cv2.erode(_mask, element)
    return _mask

def fill_holes(mask):
    mode = cv2.RETR_CCOMP
    method = cv2.CHAIN_APPROX_SIMPLE
    image, contours, hierarchy = cv2.findContours(mask, mode, method)
    out = cv2.drawContours(image, contours, -1, (1, 0, 0))

def test_clean_mask():
    mask = chip_mask
    print('Cleaning')
    mask2 = clean_mask(mask, 0, 3, .020)
    mask3 = clean_mask(mask, 3, 0, .023)
    mask4 = clean_mask(mask, 3, 3, .025)#
    mask5 = clean_mask(mask4, 2, 3, .025)
    mask6 = clean_mask(mask5, 1, 0, .025)#
    mask7 = clean_mask(mask6, 1, 0, .025)
    mask8 = clean_mask(mask7, 1, 0, .025)
    mask9 = clean_mask(mask8, 1, 3, .025)
    print('Drawing')
    df2.imshow(mask,  plotnum=331)
    df2.imshow(mask2, plotnum=332)
    df2.imshow(mask3, plotnum=333)
    df2.imshow(mask4, plotnum=334)
    df2.imshow(mask5, plotnum=335)
    df2.imshow(mask6, plotnum=336)
    df2.imshow(mask7, plotnum=337)
    df2.imshow(mask8, plotnum=338)
    df2.imshow(mask9, plotnum=339)
    print('Updating')
    df2.update()
    print('Done')

# Open CV relevant values:
# grabcut_mode = cv2.GC_EVAL
# grabcut_mode = cv2.GC_INIT_WITH_RECT
# cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_FGD


def segment(img_fpath, roi_, new_size=None):
    'Runs grabcut'
    printDBG('[segm] segment(img_fpath=%r, roi=%r)>' % (img_fpath, roi_))
    num_iters = 5
    bgd_model = np.zeros((1,13*5),np.float64)
    fgd_model = np.zeros((1,13*5),np.float64)
    mode = cv2.GC_INIT_WITH_MASK
    # Initialize
    # !!! CV2 READS (H,W) !!!
    #  WH Unsafe
    img_resz, roi_resz = resize_img_and_roi(img_fpath, roi_, new_size=new_size)
    # WH Unsafe
    (img_h, img_w) = img_resz.shape[:2]                       # Image Shape
    printDBG(' * img_resz.shape=%r' % ((img_h, img_w),))
    # WH Safe
    tlbr = algos.xywh_to_tlbr(roi_resz, (img_w, img_h))  # Rectangle ROI
    (x1, y1, x2, y2) = tlbr
    rect = tuple(roi_resz)                               # Initialize: rect 
    printDBG(' * rect=%r' % (rect,))
    printDBG(' * tlbr=%r' % (tlbr,))
    # WH Unsafe
    _mask = np.zeros((img_h,img_w), dtype=np.uint8) # Initialize: mask
    _mask[y1:y2, x1:x2] = cv2.GC_PR_FGD             # Set ROI to cv2.GC_PR_FGD 
    # Grab Cut
    tt = helpers.Timer(' * cv2.grabCut()', verbose=DEBUG_SEGM)
    cv2.grabCut(img_resz, _mask, rect, bgd_model, fgd_model, num_iters, mode=mode) 
    tt.toc()
    img_mask = np.where((_mask==cv2.GC_FGD) + (_mask==cv2.GC_PR_FGD),255,0).astype('uint8')
    # Crop 
    chip      = img_resz[y1:y2, x1:x2]
    chip_mask = img_mask[y1:y2, x1:x2]
    chip_mask = clean_mask(chip_mask)
    chip_mask = np.array(chip_mask, np.float) / 255.0
    # Mask the value of HSV
    chip_hsv = cv2.cvtColor(chip, cv2.COLOR_RGB2HSV)
    chip_hsv = np.array(chip_hsv, dtype=np.float) / 255.0
    chip_hsv[:,:,2] *= chip_mask
    chip_hsv = np.array(np.round(chip_hsv * 255.0), dtype=np.uint8)
    seg_chip = cv2.cvtColor(chip_hsv, cv2.COLOR_HSV2RGB)
    return seg_chip, img_mask

def test2(chip, chip_mask):

    im(chip, 1)
    im(chip_mask, 2)

    chip_hsv = cv2.cvtColor(chip, cv2.COLOR_RGB2HSV)
    chip_H = chip_hsv[:,:,0]
    chip_S = chip_hsv[:,:,1]
    chip_V = chip_hsv[:,:,2]

    im(chip_H, 3)
    im(chip_S, 4)
    im(chip_V, 5)

    #chip_H *= chip_mask
    #chip_S *= chip_mask
    chip_V *= chip_mask

    im(chip_V, 6)

    chip_hsv[:,:,0] = chip_H
    chip_hsv[:,:,1] = chip_S
    chip_hsv[:,:,2] = chip_V

    seg_chip = cv2.cvtColor(chip_hsv, cv2.COLOR_HSV2RGB)

    im(seg_chip, 8)
    df2.present()
    


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('[segm] __main__ = segmentation.py')
    df2.reset()
    import dev
    import helpers
    main_locals = dev.dev_main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    cx = 0
    test(hs, cx)
    #cx = int(sys.argv[1])
    exec(df2.present())
