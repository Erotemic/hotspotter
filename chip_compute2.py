from __future__ import division, print_function
import __builtin__
import sys
from PIL import Image
from Parallelize import parallel_compute
#from Printable import DynStruct
import DataStructures as ds
import helpers
import algos
#import load_data2 as ld2
import numpy as np
#import os
#import scipy.signal
#import scipy.ndimage.filters as filters
from _tpl.other import imtools

import warnings
from itertools import izip
from os.path import join

import skimage
import skimage.morphology
import skimage.filter.rank
import skimage.exposure
import skimage.util

import segmentation

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write


def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write


def print_off():
    global print, print_

    def print(*args, **kwargs):
        pass

    def print_(*args, **kwargs):
        pass


def reload_module():
    'Dynamic module reloading'
    import imp
    print('[cc2] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


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


# =======================================
# Parallelizable Work Functions
# =======================================
def __compute_chip(img_path, chip_path, roi, new_size):
    '''Crops chip from image ; Converts to grayscale ;
    Resizes to standard size ; Equalizes the histogram
    Saves as png'''
    # Read image
    img = Image.open(img_path)
    (x1, y1, x2, y2) = xywh_to_tlbr(roi, img.size)
    # http://docs.wand-py.org/en/0.3.3/guide/resizecrop.html#crop-images
    # Crop out ROI: left, upper, right, lower
    #img.transform(resize='x100') #img.transform(resize='640x480>')
    raw_chip = img.crop((x1, y1, x2, y2))
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Save chip to disk
    return chip


def rotate_chip(chip_path, rchip_path, theta):
    ''' reads chip, rotates, and saves'''
    chip = Image.open(chip_path)
    degrees = theta * 180. / np.pi
    rchip = chip.rotate(degrees, resample=Image.BICUBIC, expand=1)
    rchip.save(rchip_path, 'PNG')


# Why doesn't this work?
def make_compute_chip_func(preproc_func_list):
    def custom_compute_chip(img_path, chip_path, roi, new_size):
        chip = __compute_chip(img_path, chip_path, roi, new_size)
        for preproc_func in iter(preproc_func_list):
            print('[cc2] ' + preproc_func.__name__)
            chip = preproc_func(chip)
        chip.save(chip_path, 'PNG')
        return True
    return custom_compute_chip


def compute_grabcut_chip(img_path, chip_path, roi, new_size):
    seg_chip, img_mask = segmentation.segment(img_path, roi, new_size)
    raw_chip = Image.fromarray(seg_chip)
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    chip.save(chip_path, 'PNG')
    return True


def compute_bare_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip.save(chip_path, 'PNG')
    return True


def compute_otsu_thresh_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_reg_norm_and_histeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_reg_norm_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_histeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_contrast_stretch_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = contrast_strech(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_localeq_contr_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = local_equalize(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_localeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = local_equalize(chip)
    chip = contrast_strech(chip)
    chip.save(chip_path, 'PNG')
    return True


def compute_rankeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = rank_equalize(chip)
    chip.save(chip_path, 'PNG')
    return True

# ---------------
# Preprocessing algos


def chip_decorator(func):
    def wrapper(*arg, **kwargs):
        return func(*arg, **kwargs)
    wrapper.__name__ = 'chip_decorator_' + func.__name__


def contrast_strech(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        #p2 = np.percentile(chip_, 2)
        #p98 = np.percentile(chip_, 98)
        chip_ = skimage.exposure.equalize_hist(chip_)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip


def local_equalize(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = skimage.util.img_as_uint(chip)
        chip_ = skimage.exposure.equalize_adapthist(chip_, clip_limit=0.03)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip


def rank_equalize(chip):
    #chip_ = skimage.util.img_as_ubyte(chip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        selem = skimage.morphology.disk(30)
        chip_ = skimage.filter.rank.equalize(chip_, selem=selem)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
        return retchip


def skimage_historam_equalize(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        p2 = np.percentile(chip_, 2)
        p98 = np.percentile(chip_, 98)
        chip_ = skimage.exposure.rescale_intensity(chip_, in_range=(p2, p98))
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip


def histeq(pil_img):
    img = np.asarray(pil_img)
    return Image.fromarray(imtools.histeq(img)).convert('L')


def region_normalize_chip(chip):
    #chip = hs.get_chip(1)
    chip_ = np.asarray(chip, dtype=np.float)
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
    retchip = Image.fromarray(chip_).convert('L')
    return retchip


def pil2_float_img(chip):
    return skimage.util.img_as_float(chip)
    #chip_ = np.asarray(chip, dtype=np.float)
    #if chip_.max() > 1:
        #chip_ /= 255.0
    return chip_


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
def load_chips(hs, cx_list=None, **kwargs):
    print('\n=============================')
    print('[cc2] Precomputing chips and loading chip paths: %r' % hs.get_db_name())
    print('=============================')
    #----------------
    # COMPUTE SETUP
    #----------------
    # 1.1) Get/Update ChipConfig and ChipPaths objects
    #print('[cc2] cx_list = %r' % (cx_list,))
    if hs.prefs.chip_cfg is not None:
        hs.prefs.chip_cfg.update(**kwargs)
    else:
        hs.prefs.chip_cfg = ds.ChipConfig(**kwargs)
    chip_cfg = hs.prefs.chip_cfg
    if hs.cpaths is None:
        hs.cpaths = ds.HotspotterChipPaths()
    chip_uid = chip_cfg.get_uid()
    print('[cc2] chip_uid = %r' % chip_uid)
    if hs.cpaths.chip_uid != '' and hs.cpaths.chip_uid != chip_uid:
        print('[cc2] Disagreement: chip_uid = %r' % hs.cpaths.chip_uid)
        print('[cc2] Unloading all chip information')
        hs.unload_all()
        hs.cpaths = ds.HotspotterChipPaths()
    # Get the list of chips to load
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
    grabcut     = chip_cfg['grabcut']
    histeq      = chip_cfg['histeq']
    region_norm = chip_cfg['region_norm']
    rankeq      = chip_cfg['rank_eq']
    localeq     = chip_cfg['local_eq']
    maxcontr    = chip_cfg['maxcontrast']

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
    _rfname_fmt = 'cid%d' + chip_uid + '.rot.png'
    _rfpath_fmt = join(hs.dirs.rchip_dir, _rfname_fmt)
    rfpath_list_ = [_rfpath_fmt % cid for cid in iter(cid_list)]
    # If theta is 0 there is no need to rotate
    _fn = lambda cfpath, rfpath, theta: cfpath if theta == 0 else rfpath
    rfpath_list = [_fn(*tup) for tup in izip(cfpath_list, rfpath_list_, theta_list)]

    #--------------------------
    # EXTRACT AND RESIZE CHIPS
    #--------------------------
    num_procs = hs.args.num_procs
    if len(cx_list) < num_procs / 2:
        num_procs = 1  # Hack for small amount of tasks
    pcc_kwargs = {
        'arg_list': [gfpath_list, cfpath_list, roi_list, chipsz_list],
        'lazy': not hs.args.nocache_chips,
        'num_procs': hs.args.num_procs, }
    # FIXME: Parallel Computations of different parameters. Not robust to all parameter settings
    if grabcut:
        parallel_compute(compute_grabcut_chip, **pcc_kwargs)
    elif region_norm and histeq:
        parallel_compute(compute_reg_norm_and_histeq_chip, **pcc_kwargs)
    elif region_norm:
        parallel_compute(compute_reg_norm_chip, **pcc_kwargs)
    elif histeq:
        parallel_compute(compute_histeq_chip, **pcc_kwargs)
    elif rankeq:
        parallel_compute(compute_rankeq_chip, **pcc_kwargs)
    elif localeq and maxcontr:
        parallel_compute(compute_localeq_contr_chip, **pcc_kwargs)
    elif localeq:
        parallel_compute(compute_localeq_chip, **pcc_kwargs)
    elif maxcontr:
        parallel_compute(compute_contrast_stretch_chip, **pcc_kwargs)
    else:
        parallel_compute(compute_bare_chip, **pcc_kwargs)

    #--------------------------
    # ROTATE CHIPS
    #--------------------------
    # Get the computations that you need to do (i.e. theta != 0)
    indexes2 = [lx for lx, theta in enumerate(theta_list) if theta != 0]
    theta_list2  = [theta_list[lx] for lx in iter(indexes2)]
    cfpath_list2 = [cfpath_list[lx] for lx in iter(indexes2)]
    rfpath_list2 = [rfpath_list[lx] for lx in iter(indexes2)]

    pcc_kwargs['arg_list'] = [cfpath_list2, rfpath_list2, theta_list2]
    parallel_compute(rotate_chip, **pcc_kwargs)

    # Read sizes
    rsize_list = [(None, None) if path is None else Image.open(path).size for path in cfpath_list]
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
        hs.cpaths.cx2_rchip_path[cx] = rfpath_list[lx]
    for lx, cx in enumerate(cx_list):
        hs.cpaths.cx2_rchip_size[cx] = rsize_list[lx]
    #hs.load_cx2_rchip_size()  # TODO: Loading rchip size should be handled more robustly
    print('[cc2]=============================')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import main
    import HotSpotter
    import vizualizations as viz
    import chip_compute2 as cc2
    from chip_compute2 import *  # NOQA
    # Debugging vars
    chip_cfg = None
    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = main.parse_arguments(db='NAUTS')
    hs = HotSpotter.HotSpotter(args)
    hs.load_tables()
    hs.update_samples()
    # --- LOAD CHIPS --- #
    cc2.load_chips(hs)
    cx = helpers.get_arg_after('--cx', type_=int)
    if not cx is None:
        viz.show_chip(hs, cx, draw_kpts=False)
    else:
        print('usage: feature_compute.py --cx [cx]')
    exec(viz.present())
