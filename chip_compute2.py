from __future__ import division
from PIL import Image
from Parallelize import parallel_compute
from Printable import DynStruct
from helpers import ensure_path, mystats, myprint
import algos
import load_data2
import numpy as np
import os, sys
import params
import scipy.signal
import scipy.ndimage.filters as filters
from tpl.other import imtools

import warnings

import skimage
import skimage.morphology
import skimage.filter.rank
import skimage.exposure
import skimage.util

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

# =======================================
# Parallelizable Work Functions          
# =======================================
def __compute_chip(img_path, chip_path, roi, new_size):
    '''Crops chip from image ; Converts to grayscale ; 
    Resizes to standard size ; Equalizes the histogram
    Saves as png'''
    # Read image
    img = Image.open(img_path)
    [img_w, img_h] = [ gdim - 1 for gdim in img.size ]
    # Ensure ROI is within bounds
    [roi_x, roi_y, roi_w, roi_h] = [max(0, cdim) for cdim in roi]
    roi_x2 = min(img_w, roi_x + roi_w)
    roi_y2 = min(img_h, roi_y + roi_h)
    # http://docs.wand-py.org/en/0.3.3/guide/resizecrop.html#crop-images
    # Crop out ROI: left, upper, right, lower
    #img.transform(resize='x100')
    #img.transform(resize='640x480>')
    raw_chip = img.crop((roi_x, roi_y, roi_x2, roi_y2))
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
            print preproc_func.__name__
            chip = preproc_func(chip)
        chip.save(chip_path, 'PNG')
        return True
    return custom_compute_chip

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
    wrapper.__name__ = 'chip_decorator_'+func.__name__

def contrast_strech(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        p2 = np.percentile(chip_, 2)
        p98 = np.percentile(chip_, 98)
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
    x1 = chipw/2 - half_w
    y1 = chiph/2 - half_h
    x2 = chipw/2 + half_w
    y2 = chiph/2 + half_h
    (x1,y1,x2,y2) = map(int, map(round, (x1,y1,x2,y2)))
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

# =======================================
# Main Script 
# =======================================

class HotspotterChipPaths(DynStruct):
    def __init__(self):
        super(HotspotterChipPaths, self).__init__()
        self.cx2_chip_path  = []
        self.cx2_rchip_path = []

def load_chip_paths(hs_dirs, hs_tables, hs=None):
    img_dir      = hs_dirs.img_dir
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir

    cx2_gx       = hs_tables.cx2_gx
    cx2_cid      = hs_tables.cx2_cid
    cx2_theta    = hs_tables.cx2_theta
    cx2_roi      = hs_tables.cx2_roi
    gx2_gname    = hs_tables.gx2_gname

    print('=============================')
    print('cc2> Precomputing chips and loading chip paths')
    print('=============================')

    # Get parameters
    sqrt_area   = params.__CHIP_SQRT_AREA__
    histeq      = params.__HISTEQ__
    region_norm = params.__REGION_NORM__
    rankeq     = params.__RANK_EQ__
    localeq    = params.__LOCAL_EQ__
    maxcontr    = params.__MAXCONTRAST__

    chip_params = dict(sqrt_area=sqrt_area, histeq=histeq)
    chip_uid = params.get_chip_uid()

    print(' * chip_uid    = %r' % chip_uid)

    print(' * sqrt(target_area) = %r' % sqrt_area)
    print(' * localeq     = %r' % localeq)
    print(' * maxcontr    = %r' % maxcontr)
    print(' * histeq      = %r' % histeq)
    print(' * rankeq      = %r' % rankeq)
    print(' * region_norm = %r' % region_norm)

    # Full image path
    cx2_img_path = [img_dir+'/'+gx2_gname[gx] for gx in cx2_gx]
    # Paths to chip, rotated chip
    chip_format  =  chip_dir+'/CID_%d'+chip_uid+'.png'
    rchip_format = rchip_dir+'/CID_%d'+chip_uid+'.rot.png'
    cx2_chip_path   = [chip_format  % cid for cid in cx2_cid]
    cx2_rchip_path  = [rchip_format % cid for cid in cx2_cid]
    # Normalized chip size
    cx2_imgchip_sz = [(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    if not (sqrt_area is None or sqrt_area <= 0):
        target_area = sqrt_area ** 2
        def _resz(w, h):
            ht = np.sqrt(target_area * h / w)
            wt = w * ht / h
            return (int(round(wt)), int(round(ht)))
        cx2_chip_sz = [_resz(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    else: # no rescaling
        cx2_chip_sz = [(int(w), int(h)) for (x,y,w,h) in cx2_roi]

    # --- COMPUTE CHIPS --- # 
    pcc_kwargs = {
        'arg_list'  : [cx2_img_path, cx2_chip_path, cx2_roi, cx2_chip_sz],
        'lazy'      : (not '--nochipcache' in sys.argv),
        'num_procs' : params.__NUM_PROCS__ }
    #(img_path, chip_path, roi, new_size) = zip(*pcc_kwargs['arg_list'])[109]
    # Make a compute chip function with all the preprocessing your heart desires
    #preproc_func_list = []
    #if region_norm: 
        #preproc_func_list.append(region_normalize_chip)
    #if histeq: 
        #preproc_func_list.append(algos.histeq)
    #custom_compute_chip = make_compute_chip_func(preproc_func_list)

    if region_norm and histeq: 
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

    # --- ROTATE CHIPS --- # 
    pcc_kwargs['arg_list'] = [cx2_chip_path, cx2_rchip_path, cx2_theta] 
    parallel_compute(rotate_chip, **pcc_kwargs)

    # --- RETURN CHIP PATHS --- #

    print('Done Precomputing chips and loading chip paths')

    # Build hotspotter path object
    hs_cpaths = HotspotterChipPaths()
    hs_cpaths.cx2_chip_path  = cx2_chip_path
    hs_cpaths.cx2_rchip_path = cx2_rchip_path
    
    if not hs is None:
        hs.cpaths = hs_cpaths

    return hs_cpaths

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import load_data2
    freeze_support()
    # --- LOAD DATA --- #
    db_dir = load_data2.DEFAULT
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    hs = load_data2.HotSpotter(None)
    hs.tables = hs_tables
    hs.dirs   = hs_dirs
    # --- LOAD CHIPS --- #
    hs_cpaths = load_chip_paths(hs_dirs, hs_tables)
