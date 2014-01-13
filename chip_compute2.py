from __future__ import division, print_function
import __common__
(print, print_,
 print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[cc2]')
# Python
import sys
import warnings
from os.path import join
# Science
import numpy as np
import cv2
from PIL import Image
# Hotspotter
import helpers
import fileio as io
from Parallelize import parallel_compute
from _tpl.other import imtools
#from Printable import DynStruct
#import load_data2 as ld2
#import os
#import scipy.signal
#import scipy.ndimage.filters as filters
#import skimage
#import skimage.morphology
#import skimage.filter.rank
#import skimage.exposure
#import skimage.util


DEBUG = False
if DEBUG:
    def printDBG(msg):
        print(msg)
        sys.stdout.flush()
else:
    def printDBG(msg):
        pass


def xywh_to_tlbr(roi, img_wh):
    (img_w, img_h) = img_wh
    if img_w == 0 or img_h == 0:
        msg = '[cc2.1] Your csv tables have an invalid ROI.'
        print(msg)
        warnings.warn(msg)
        img_w = 1
        img_h = 1
    # Ensure ROI is within bounds
    (x, y, w, h) = roi
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_w - 1)
    y2 = min(y + h, img_h - 1)
    return (x1, y1, x2, y2)


def rgb_to_gray(rgb_img):
    pil_gray_img = Image.fromarray(rgb_img).convert('L')
    gray_img = np.asarray(pil_gray_img)
    return gray_img


def gray_to_rgb(gray_img):
    rgb_img = np.empty(list(gray_img.shape) + [3], gray_img.dtype)
    rgb_img[:, :, 0] = gray_img
    rgb_img[:, :, 1] = gray_img
    rgb_img[:, :, 2] = gray_img
    return rgb_img


def ensure_gray(img):
    'Ensures numpy format and 3 channels'
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return img


def ensure_rgb(img):
    try:
        'Ensures numpy format and 3 channels'
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        if img.dtype == np.float64 or np.dtype == np.float32:
            if img.max() <= 1:
                img *= 255
            img = np.array(np.round(img), dtype=np.uint8)
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Only given 1 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    except Exception as ex:
        msg = ('[cc2] Caught Exception:\n   ex=%s\n' % str(ex) +
               '[cc2] img.shape=%r, img.dtype=%r\n' % (img.shape, img.dtype) +
               '[cc2] stats(img) = %s' % (helpers.printable_mystats(img)))
        print(msg)
        raise Exception(msg)

# =======================================
# Parallelizable Work Functions
# =======================================
NEW_ORIENT = True
if NEW_ORIENT:
    def build_transform(x, y, w, h, w_, h_, theta):
        sx = (w_ / w)  # ** 2
        sy = (h_ / h)  # ** 2
        cos_ = np.cos(-theta)
        sin_ = np.sin(-theta)
        tx = -(x + (w / 2))
        ty = -(y + (h / 2))
        T1 = np.array([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0, 1]], np.float64)

        S = np.array([[sx, 0,  0],
                      [0, sy,  0],
                      [0,  0,  1]], np.float64)

        R = np.array([[cos_, -sin_, 0],
                      [sin_,  cos_, 0],
                      [   0,     0, 1]], np.float64)

        T2 = np.array([[1, 0, (w_ / 2)],
                       [0, 1, (h_ / 2)],
                       [0, 0, 1]], np.float64)
        M = T2.dot(R.dot(S.dot(T1)))
        #.dot(R)#.dot(S).dot(T2)
        Aff = M[0:2, :] / M[2, 2]
        #helpers.horiz_print(S, T1, T2)
        #print('T1======')
        #print(T1)
        #print('R------')
        #print(R)
        #print('S------')
        #print(S)
        #print('T2------')
        #print(T2)
        #print('M------')
        #print(M)
        #print('Aff------')
        #print(Aff)
        #print('======')
        return Aff

#cv2_flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
#cv2_borderMode  = cv2.BORDER_CONSTANT
#cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}


def extract_chip(img_path, chip_path, roi, theta, new_size):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    #printDBG('[cc2] reading image')
    np_img = io.imread(img_path)
    #printDBG('[cc2] building transform')
    # Build transformation
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    Aff = build_transform(rx, ry, rw, rh, rw_, rh_, theta)
    #printDBG('[cc2] rotate and scale')
    # Rotate and scale
    flags = cv2.INTER_LINEAR
    borderMode = cv2.BORDER_CONSTANT
    chip = cv2.warpAffine(np_img, Aff, (rw_, rh_), flags=flags, borderMode=borderMode)
    #printDBG('[cc2] return extracted')
    return chip


def compute_chip(img_path, chip_path, roi, theta, new_size, filter_list):
    '''Extracts Chip; Applies Filters; Saves as png'''
    #printDBG('[cc2] extracting chip')
    chip = extract_chip(img_path, chip_path, roi, theta, new_size)
    #printDBG('[cc2] extracted chip')
    for func in filter_list:
        #printDBG('[cc2] computing filter: %r' % func)
        chip = func(chip)
    # Convert to grayscale
    pil_chip = Image.fromarray(chip).convert('L')
    #printDBG('[cc2] saving chip: %r' % chip_path)
    pil_chip.save(chip_path, 'PNG')
    #printDBG('[cc2] returning')
    return True

# ---------------
# Preprocessing funcs


def grabcut_fn(chip):
    import segmentation
    rgb_chip = ensure_rgb(chip)
    seg_chip = segmentation.grabcut(rgb_chip)
    return seg_chip


#def maxcontr_fn(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        ##p2 = np.percentile(chip_, 2)
        ##p98 = np.percentile(chip_, 98)
        #chip_ = skimage.exposure.equalize_hist(chip_)
        #retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    #return retchip


#def localeq_fn(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = skimage.util.img_as_uint(chip)
        #chip_ = skimage.exposure.equalize_adapthist(chip_, clip_limit=0.03)
        #retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    #return retchip


#def rankeq_fn(chip):
    ##chip_ = skimage.util.img_as_ubyte(chip)
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        #selem = skimage.morphology.disk(30)
        #chip_ = skimage.filter.rank.equalize(chip_, selem=selem)
        #retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
        #return retchip


#def skimage_historam_equalize(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        #p2 = np.percentile(chip_, 2)
        #p98 = np.percentile(chip_, 98)
        #chip_ = skimage.exposure.rescale_intensity(chip_, in_range=(p2, p98))
        #retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    #return retchip


def histeq_fn(chip):
    chip = ensure_gray(chip)
    chip = imtools.histeq(chip)
    return chip


def region_norm_fn(chip):
    import algos
    chip  = ensure_gray(chip)
    chip_ = np.array(chip, dtype=np.float)
    chipw, chiph = chip_.shape
    half_w = chipw * .1
    half_h = chiph * .1
    x1 = chipw / 2 - half_w
    y1 = chiph / 2 - half_h
    x2 = chipw / 2 + half_w
    y2 = chiph / 2 + half_h
    (x1, y1, x2, y2) = map(int, map(round, (x1, y1, x2, y2)))
    area = chip_[x1:x2, y1:y2]
    intensity = area.flatten()
    freq, _  = np.histogram(intensity, 32)
    #algos.viz_localmax(freq)
    maxpos  = algos.localmax(freq)
    min_int = intensity.min()
    max_int = intensity.max()
    maxima = min_int + (max_int - min_int) * np.array(maxpos) / float(len(freq))
    if len(maxima) > 2:
        low  = float(maxima[0])
        high = float(maxima[-1])
    else:
        low  = float(min_int)
        high = float(max_int)
    chip_ = (chip_ - low) * 255.0 / (high - low)
    chip_ = chip_.round()
    chip_[chip_ < 0] = 0
    chip_[chip_ > 255] = 255
    return chip_


#def pil2_float_img(chip):
    #return skimage.util.img_as_float(chip)
    ##chip_ = np.asarray(chip, dtype=np.float)
    ##if chip_.max() > 1:
        ##chip_ /= 255.0
    #return chip_


def get_normalized_chip_sizes(roi_list, sqrt_area=None):
    'Computes a normalized chip size to rescale to'
    if not (sqrt_area is None or sqrt_area <= 0):
        target_area = sqrt_area ** 2

        def _resz(w, h):
            try:
                ht = np.sqrt(target_area * h / w)
                wt = w * ht / h
                return (int(round(wt)), int(round(ht)))
            except Exception:
                msg = '[cc2.2] Your csv tables have an invalid ROI.'
                print(msg)
                warnings.warn(msg)
                return (1, 1)
        chipsz_list = [_resz(float(w), float(h)) for (x, y, w, h) in roi_list]
    else:  # no rescaling
        chipsz_list = [(int(w), int(h)) for (x, y, w, h) in roi_list]
    return chipsz_list


# =======================================
# Main Script
# =======================================
#@profile
def load_chips(hs, cx_list=None, **kwargs):
    print('\n=============================')
    print('[cc2] Precomputing chips and loading chip paths: %r' % hs.get_db_name())
    print('=============================')
    #----------------
    # COMPUTE SETUP
    #----------------
    chip_cfg = hs.prefs.chip_cfg
    chip_uid = chip_cfg.get_uid()
    if hs.cpaths.chip_uid != '' and hs.cpaths.chip_uid != chip_uid:
        print('[cc2] Disagreement: OLD_chip_uid = %r' % hs.cpaths.chip_uid)
        print('[cc2] Disagreement: NEW_chip_uid = %r' % chip_uid)
        print('[cc2] Unloading all chip information')
        hs.unload_all()
    print('[cc2] chip_uid = %r' % chip_uid)
    # Get the list of chips paths to load
    cx_list = hs.get_valid_cxs() if cx_list is None else cx_list
    if not np.iterable(cx_list):
        cx_list = [cx_list]
    if len(cx_list) == 0:
        return  # HACK
    cx_list = np.array(cx_list)  # HACK
    hs.cpaths.chip_uid = chip_uid
    #print('[cc2] Requested %d chips' % (len(cx_list)))
    #print('[cc2] cx_list = %r' % (cx_list,))
    # Get table information
    try:
        gx_list    = hs.tables.cx2_gx[cx_list]
        cid_list   = hs.tables.cx2_cid[cx_list]
        theta_list = hs.tables.cx2_theta[cx_list]
        roi_list   = hs.tables.cx2_roi[cx_list]
        gname_list = hs.tables.gx2_gname[gx_list]
    except IndexError as ex:
        print(repr(ex))
        print(hs.tables)
        print('cx_list=%r' % (cx_list,))
        raise
    # Get ChipConfig Parameters
    sqrt_area   = chip_cfg['chip_sqrt_area']

    filter_list = []
    if chip_cfg['histeq']:
        filter_list.append(histeq_fn)
    if chip_cfg['region_norm']:
        filter_list.append(region_norm_fn)
    if chip_cfg['maxcontrast']:
        filter_list.append(maxcontr_fn)
    if chip_cfg['rank_eq']:
        filter_list.append(rankeq_fn)
    if chip_cfg['local_eq']:
        filter_list.append(localeq_fn)
    if chip_cfg['grabcut']:
        filter_list.append(grabcut_fn)

    #---------------------------
    # ___Normalized Chip Args___
    #---------------------------
    # Full Image Paths: where to extract the chips from
    img_dir = hs.dirs.img_dir
    gfpath_list = [join(img_dir, gname) for gname in iter(gname_list)]
    # Chip Paths: where to write extracted chips to
    _cfname_fmt = 'cid%d' + chip_uid + '.png'
    _cfpath_fmt = join(hs.dirs.chip_dir, _cfname_fmt)
    cfpath_list = [_cfpath_fmt  % cid for cid in iter(cid_list)]
    # Normalized Chip Sizes: ensure chips have about sqrt_area squared pixels
    chipsz_list = get_normalized_chip_sizes(roi_list, sqrt_area)

    #-------------------------
    #____Rotated Chip Args____
    #-------------------------
    # Rotated Chp Paths: where to write rotated chips to
    #_rfname_fmt = 'cid%d' + chip_uid + '.rot.png'
    #_rfpath_fmt = join(hs.dirs.rchip_dir, _rfname_fmt)
    #rfpath_list_ = [_rfpath_fmt % cid for cid in iter(cid_list)]
    # If theta is 0 there is no need to rotate
    #_fn = lambda cfpath, rfpath, theta: cfpath if theta == 0 else rfpath
    #rfpath_list = [_fn(*tup) for tup in izip(cfpath_list, rfpath_list_, theta_list)]

    #--------------------------
    # EXTRACT AND RESIZE CHIPS
    #--------------------------
    pcc_kwargs = {
        'arg_list': [gfpath_list, cfpath_list, roi_list, theta_list, chipsz_list],
        'lazy': not hs.args.nocache_chips,
        'num_procs': hs.args.num_procs,
        'common_args': [filter_list]
    }
    # Compute all chips with paramatarized filters
    parallel_compute(compute_chip, **pcc_kwargs)

    # Read sizes
    try:
        rsize_list = [(None, None) if path is None else Image.open(path).size
                      for path in iter(cfpath_list)]
    except IOError as ex:
        import gc
        gc.collect()
        print('[cc] ex=%r' % ex)
        print('path=%r' % path)
        if helpers.checkpath(path, verbose=True):
            import time
            time.sleep(1)  # delays for 1 seconds
            print('[cc] file exists but cause IOError?')
            print('[cc] probably corrupted. Removing it')
            try:
                helpers.remove_file(path)
            except OSError:
                print('Something bad happened')
                raise
        raise
    #----------------------
    # UPDATE API VARIABLES
    #----------------------
    print('[cc2] Done Precomputing chips and loading chip paths')

    # Extend the datastructure if needed
    list_size = max(cx_list) + 1
    #helpers.ensure_list_size(hs.cpaths.cx2_chip_path, list_size)
    helpers.ensure_list_size(hs.cpaths.cx2_rchip_path, list_size)
    helpers.ensure_list_size(hs.cpaths.cx2_rchip_size, list_size)
    # Copy the values into the ChipPaths object
    #for lx, cx in enumerate(cx_list):
        #hs.cpaths.cx2_chip_path[cx] = cfpath_list[lx]
    for lx, cx in enumerate(cx_list):
        hs.cpaths.cx2_rchip_path[cx] = cfpath_list[lx]
    for lx, cx in enumerate(cx_list):
        hs.cpaths.cx2_rchip_size[cx] = rsize_list[lx]
    #hs.load_cx2_rchip_size()  # TODO: Loading rchip size should be handled more robustly
    print('[cc2]=============================')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import HotSpotter
    import argparse2
    import vizualizations as viz
    import chip_compute2 as cc2
    from chip_compute2 import *  # NOQA
    # Debugging vars
    chip_cfg = None
#l')=103.7900s

    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = argparse2.parse_arguments(defaultdb='NAUTS')
    hs = HotSpotter.HotSpotter(args)
    hs.load_tables()
    hs.update_samples()
    # --- LOAD CHIPS --- #
    cc2.load_chips(hs)
    cx = helpers.get_arg_after('--cx', type_=int)
    if not cx is None:
        tau = np.pi * 2
        hs.change_theta(cx, tau / 8)
        viz.show_chip(hs, cx, draw_kpts=False, fnum=1)
        viz.show_image(hs, hs.cx2_gx(cx), fnum=2)
    else:
        print('usage: feature_compute.py --cx [cx]')
    exec(viz.df2.present())
