from __future__ import division
from PIL import Image
from hotspotter.Parallelize import parallel_compute
from hotspotter.algo.imalgos import histeq
from hotspotter.helpers import ensure_path, mystats, myprint
import load_data2
import numpy as np
import os, sys
import feature_compute2 as fc2

__DBG_INFO__ = False

# =======================================
# Parallelizable Work Functions          
# =======================================
def compute_chip(img_path, chip_path, roi, new_size):
    ''' 
    Crops chip from image
    Converts to grayscale
    Resizes to standard size
    Equalizes the histogram
    Saves as png
    '''
    # Read image
    img = Image.open(img_path)
    [img_w, img_h] = [ gdim - 1 for gdim in img.size ]
    # Ensure ROI is within bounds
    [roi_x, roi_y, roi_w, roi_h] = [ max(0, cdim) for cdim in roi]
    roi_x2 = min(img_w, roi_x + roi_w)
    roi_y2 = min(img_h, roi_y + roi_h)
    # Crop out ROI: left, upper, right, lower
    raw_chip = img.crop((roi_x, roi_y, roi_x2, roi_y2))
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Preprocessing based on preferences
    chip = histeq(chip)
    # Save chip to disk
    chip.save(chip_path, 'PNG')
    return True

def rotate_chip(chip_path, rchip_path, theta):
    ''' reads chip, rotates, and saves'''
    chip = Image.open(chip_path)
    degrees = theta * 180. / np.pi
    rchip = chip.rotate(degrees, resample=Image.BICUBIC, expand=1)
    rchip.save(rchip_path, 'PNG')

# =======================================
# Main Script 
# =======================================

if __name__ == '__main__':
    # <Support for windows>
    from multiprocessing import freeze_support
    freeze_support()
    # </Support for windows>

    # --- LOAD DATA --- #
    db_dir = load_data2.MOTHERS
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # These are bad, but convinient. 
    #print(hs_tables.execstr('hs_tables'))
    #exec(hs_tables.execstr('hs_tables'))
    #print(hs_dirs.execstr('hs_dirs'))
    #exec(hs_dirs.execstr('hs_dirs'))
    #print(hs_tables)
    #print(hs_dirs)
    name_table   = hs_dirs.name_table
    feat_dir     = hs_dirs.feat_dir
    chip_table   = hs_dirs.chip_table
    img_dir      = hs_dirs.img_dir
    internal_sym = hs_dirs.internal_sym
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir
    image_table  = hs_dirs.image_table
    internal_dir = hs_dirs.internal_dir
    db_dir       = hs_dirs.db_dir

    px2_propname = hs_tables.px2_propname
    px2_cx2_prop = hs_tables.px2_cx2_prop
    cx2_gx = hs_tables.cx2_gx
    cx2_cid = hs_tables.cx2_cid
    nx2_name = hs_tables.nx2_name
    cx2_nx = hs_tables.cx2_nx
    cx2_theta = hs_tables.cx2_theta
    cx2_roi = hs_tables.cx2_roi
    gx2_gname = hs_tables.gx2_gname
    
    # --- CREATE COMPUTED DIRS --- #

    # --- BUILD TASK INFORMATION --- #
    # Full image path
    cx2_img_path  = [ img_dir+'/'+gx2_gname[gx]   for gx  in cx2_gx ]
  
    # Paths to chip, rotated chip, and chip features 
    cx2_chip_path   = [ chip_dir+'/CID_%d.png'     % cid for cid in cx2_cid]
    cx2_rchip_path  = [rchip_dir+'/CID_%d.rot.png' % cid for cid in cx2_cid]
    cx2_hesaff_path = [ feat_dir+'/CID_%d_hesaff.npz' % cid for cid in cx2_cid]
    cx2_sift_path   = [ feat_dir+'/CID_%d_sift.npz'   % cid for cid in cx2_cid]
    cx2_freak_path  = [ feat_dir+'/CID_%d_freak.npz'  % cid for cid in cx2_cid]

    # Normalized chip size
    At = 500.0 ** 2 # target area
    def _target_resize(w, h):
        ht = np.sqrt(At * h / w)
        wt = w * ht / h
        return (int(round(wt)), int(round(ht)))
    cx2_chip_sz = [_target_resize(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    cx2_imgchip_sz = [(float(w), float(h)) for (x,y,w,h) in cx2_roi]

    if __DBG_INFO__:
        cx2_chip_sf = [(w/wt, h/ht)
                    for ((w,h),(wt,ht)) in zip(cx2_imgchip_sz, cx2_chip_sz)]
        cx2_sf_ave = [(sf1 + sf2) / 2 for (sf1, sf2) in cx2_chip_sf]
        cx2_sf_err = [np.abs(sf1 - sf2) for (sf1, sf2) in cx2_chip_sf]
        myprint(mystats(cx2_sf_ave),lbl='ave scale factor')
        myprint(mystats(cx2_sf_err),lbl='ave scale factor error')

    # --- COMPUTE CHIPS --- # 
    parallel_compute(compute_chip,
                     arg_list=[cx2_img_path,
                               cx2_chip_path,
                               cx2_roi,
                               cx2_chip_sz])
    # --- ROTATE CHIPS --- # 
    parallel_compute(rotate_chip,
                     arg_list=[cx2_chip_path,
                               cx2_rchip_path,
                               cx2_theta])

    # --- COMPUTE FEATURES --- # 
    # Hessian Affine Features
    print('Computing features')
    #parallel_compute(fc2.compute_hesaff, [cx2_rchip_path, cx2_hesaff_path])
    parallel_compute(fc2.compute_sift,   [cx2_rchip_path, cx2_sift_path])
    #parallel_compute(fc2.compute_freak,  [cx2_rchip_path, cx2_freak_path]) 
    # --- LOAD FEATURES --- # 
    print('Loading features')
    #cx2_hesaff_feats = parallel_compute(fc2.load_features, [cx2_hesaff_path])
    cx2_sift_feats  = parallel_compute(fc2.load_features, [cx2_sift_path], 1)
    #cx2_sift_kpts, cx2_sift_desc = \
    #     ([ k for k,d in cx2_sift_feats ], [ d for k,d in cx2_sift_feats ])
    #cx2_freak_feats   = parallel_compute(fc2.load_features, [cx2_freak_path])
