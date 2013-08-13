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


# =======================================
# Parallelizable Work Functions          
# =======================================
def precompute_chip_bare(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip.save(chip_path, 'PNG')
    return True

# Preprocessing based on preferences
def precompute_chip_histeq(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = algos.histeq(chip)
    chip.save(chip_path, 'PNG')
    return True

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

# =======================================
# Main Script 
# =======================================

class HotspotterChipPaths(DynStruct):
    def __init__(self):
        super(HotspotterChipPaths, self).__init__()
        self.cx2_chip_path  = []
        self.cx2_rchip_path = []

def load_chip_paths(hs_dirs, hs_tables):
    img_dir      = hs_dirs.img_dir
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir

    cx2_gx       = hs_tables.cx2_gx
    cx2_cid      = hs_tables.cx2_cid
    cx2_theta    = hs_tables.cx2_theta
    cx2_roi      = hs_tables.cx2_roi
    gx2_gname    = hs_tables.gx2_gname

    print('=============================')
    print('Precomputing chips and loading chip paths')
    print('=============================')
    
    # --- BUILD TASK INFORMATION --- #
    ''' TODO: These should be functions
    Maybe you can change them to objects so they work like lists but dont 
    use up so much memory. Make them more like indexable generators'''
    # Get parameters
    sqrt_area = params.__CHIP_SQRT_AREA__
    histeq    = params.__HISTEQ__
    chip_params = dict(sqrt_area=sqrt_area, histeq=histeq)
    print(' * sqrt(target_area) = %r' % sqrt_area)
    print(' * histeq = %r' % histeq)

    chip_uid = params.get_chip_uid()
    # Full image path
    cx2_img_path = [img_dir+'/'+gx2_gname[gx] for gx in cx2_gx]
    # Paths to chip, rotated chip
    chip_format  = chip_dir+'/CID_%d'+chip_uid+'.png'
    rchip_format = rchip_dir+'/CID_%d'+chip_uid+'.rot.png'
    cx2_chip_path   = [chip_format % cid for cid in cx2_cid]
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
    if histeq:
        compute_chip = compute_chip_histeq
    else:
        compute_chip = precompute_chip_bare
    parallel_compute(precompute_chip_bare, arg_list=[cx2_img_path, cx2_chip_path,
                                             cx2_roi, cx2_chip_sz])
    # --- ROTATE CHIPS --- # 
    parallel_compute(rotate_chip, arg_list=[cx2_chip_path,
                                            cx2_rchip_path, cx2_theta])
    # --- RETURN CHIP PATHS --- #
    hs_cpaths = HotspotterChipPaths()
    hs_cpaths.cx2_chip_path  = cx2_chip_path
    hs_cpaths.cx2_rchip_path = cx2_rchip_path
    print('=============================')
    print('Done Precomputing chips and loading chip paths')
    print('=============================\n')

    return hs_cpaths

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # --- LOAD DATA --- #
    db_dir = load_data2.DEFAULT
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # --- LOAD CHIPS --- #
    hs_cpaths = load_chip_paths(hs_dirs, hs_tables)

# GRAVEYARD
'''
    __DBG_INFO__ = False

    if __DBG_INFO__:
        cx2_chip_sf = [(w/wt, h/ht)
                    for ((w,h),(wt,ht)) in zip(cx2_imgchip_sz, cx2_chip_sz)]
        cx2_sf_ave = [(sf1 + sf2) / 2 for (sf1, sf2) in cx2_chip_sf]
        cx2_sf_err = [np.abs(sf1 - sf2) for (sf1, sf2) in cx2_chip_sf]
        myprint(mystats(cx2_sf_ave),lbl='ave scale factor')
        myprint(mystats(cx2_sf_err),lbl='ave scale factor error')
'''
