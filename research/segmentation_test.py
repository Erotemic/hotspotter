#from __init__ import *
import numpy as np
import cv2
import draw_func2 as df2


def test():
    (hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2) = ld2.get_sv_test_data()
    # READ IMAGE AND ROI
    cx2_roi = hs.tables.cx2_roi
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname
    #---
    roi = cx2_roi[cx]
    gx  = cx2_gx[cx]
    img_fname = gx2_gname[gx]
    img_path = os.path.join(hs.dirs.img_dir, img_fname)
    #---
    #print('Showing Input')
    ## Original Image
    #df2.imshow(mask_input, fignum=1, plotnum=122)
    #df2.present()
    # Crop out chips
    #chip      = img[ry:(ry+rh), rx:(rx+rw)]
    #chip_mask = mask[ry:(ry+rh), rx:(rx+rw)]

    # IMAGE AND MASK
    #df2.imshow(img,      plotnum=121, fignum=2, doclf=True)
    #df2.imshow(mask,     plotnum=122, fignum=2)

    # MASKED CHIP
    #df2.figure(fignum=3, doclf=True)
    #df2.imshow(chip,       plotnum=131, fignum=3, doclf=True)
    #df2.imshow(color_mask, plotnum=132, fignum=3)
    #df2.imshow(seg_chip,   plotnum=133, fignum=3)

    seg_chip = segment(img_path, roi)

    df2.show_img(hs, cx, fignum=1, plotnum=121, title='original', doclf=True)
    df2.imshow(seg_chip, fignum=1, plotnum=122, title='segmented')
    df2.present()


#grabcut_mode = cv2.GC_EVAL
#grabcut_mode = cv2.GC_INIT_WITH_RECT
def segment(img_path, roi):
    img = cv2.imread(img_path)
    # Grab cut parametrs
    (w, h) = img.shape[:2]
    [rx, ry, rw, rh] = roi
    expand = 10
    rect = (rx-expand, rw+expand, ry-expand, rh+expand)
    mask_input = np.zeros((w,h), dtype=np.uint8)
    bg_model = np.zeros((1, 13 * 5))
    fg_model = np.zeros((1, 13 * 5))
    num_iters = 5
    # Make mask specify the ROI
    mask_input[ry:(ry+rh), rx:(rx+rw)] = cv2.GC_PR_FGD
    grabcut_mode = cv2.GC_INIT_WITH_MASK
    # Do grab cut
    (mask, bg_model, fg_model) = cv2.grabCut(img, mask_input, rect,
                                            bg_model, fg_model,
                                            num_iters, mode=grabcut_mode)

    # Make the segmented chip
    # Normalize mask
    mask = np.array(mask, np.float64)
    mask[mask == cv2.GC_BGD] = 0#.1
    mask[mask == cv2.GC_PR_BGD] = 0#.25
    mask[mask == cv2.GC_PR_FGD] = 1#.75
    mask[mask == cv2.GC_FGD] = 1


    # Clean the mask
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
    tmp_ = mask
    tmp_ = cv2.erode(tmp_, element)
    tmp_ = cv2.dilate(tmp_, element)
    tmp_ = cv2.dilate(tmp_, element)
    tmp_ = cv2.dilate(tmp_, element)
    df2.imshow(tmp_); df2.update()
    mask = tmp_

    # Reshape chip_mask to NxMx3
    color_mask = np.tile(chip_mask.reshape(tuple(list(chip_mask.shape)+[1])), (1, 1, 3))

    seg_chip = np.array(chip, dtype=np.float) * color_mask
    seg_chip = np.array(np.round(seg_chip), dtype=np.uint8)  
    return seg_chip

