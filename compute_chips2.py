from __future__ import division
import load_data2
from hotspotter.helpers import ensure_path, mystats, myprint
from hotspotter.Parallelize import parallelize_tasks
from PIL import Image
import os
import numpy as np

from hotspotter.algo.imalgos import histeq

__DBG_INFO__ = True

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
    hs_tables = load_data2.load_csv_tables(db_dir)
    exec(hs_tables.execstr('hs_tables'))
    print(hs_tables)

    # --- CREATE COMPUTED DIRS --- #
    img_dir  = db_dir + '/images'
    chip_dir = db_dir + '/.hs_internals/computed/chips'
    rchip_dir = db_dir + '/.hs_internals/computed/temp'
    feat_dir = db_dir + '/.hs_internals/computed/feats'
    internal_dir = db_dir + '/.hs_internals'
    internal_sym = db_dir + '/Shortcut-to-hs_internals'

    ensure_path(internal_dir)
    ensure_path(chip_dir)
    ensure_path(feat_dir)
    if not os.path.islink(internal_sym):
        os.symlink(internal_dir, internal_sym)

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
    print('Computing chips')
    compute_chip_args = iter(zip(cx2_img_path, cx2_chip_path, cx2_roi, cx2_chip_sz))
    compute_chip_tasks = [(compute_chip, _args) for _args in compute_chip_args]
    #parallelize_tasks(compute_chip_tasks, num_procs=8)

    # --- ROTATE CHIPS --- # 
    print('Rotating chips')
    rotate_chip_args = iter(zip(cx2_chip_path, cx2_rchip_path, cx2_theta))
    rotate_chip_tasks = [(rotate_chip, _args) for _args in rotate_chip_args]
    #parallelize_tasks(rotate_chip_tasks, num_procs=8)

    # --- COMPUTE FEATURES --- # 
    # Hessian Affine Features
    print('Computing hessian affine features')
    from hotspotter.tpl.hesaff import compute_hesaff
    compute_hesaff_args = iter(zip(cx2_rchip_path, cx2_hesaff_path))
    compute_hesaff_tasks = [(compute_hesaff, _args) for _args in compute_hesaff_args]
    #parallelize_tasks(compute_hesaff_tasks, num_procs=8)

    import hotspotter.tpl.cv2 as cv2
    sift_detector  = cv2.FeatureDetector_create('SURF')
    sift_extractor = cv2.DescriptorExtractor_create('SIFT')
    freak_extractor = cv2.DescriptorExtractor_create('FREAK')
    freak_extractor.setBool('orientationNormalized', False)
    ####
    def compute_sift(rchip_fpath, chiprep_fpath):
        rchip = cv2.imread(rchip_fpath)
        _cv_kpts = sift_detector.detect(rchip)  
        # gravity vector
        for cv_kp in iter(_cv_kpts): cv_kp.angle = 0
        cv_kpts, cv_descs = sift_extractor.compute(rchip, _cv_kpts)
        kpts = np.zeros((len(cv_kpts), 5), dtype=np.float32)
        desc = np.array(cv_descs, dtype=np.uint8)
        fx = 0
        for cv_kp in cv_kpts:
            (x,y) = cv_kp.pt
            theta = cv_kp.angle
            scale = float(cv_kp.size)**2 / 27
            detA  = 1./(scale)
            (a,c,d) = (detA, 0, detA)
            kpts[fx] = (x,y,a,c,d)
            fx += 1
        np.savez(chiprep_fpath, kpts, desc)
        return True
    def compute_freak(rchip_fpath, chiprep_fpath):
        rchip = cv2.imread(rchip_fpath)
        _cv_kpts = sift_detector.detect(rchip)  
        # gravity vector
        for cv_kp in iter(_cv_kpts): cv_kp.angle = 0
        cv_kpts, cv_descs = freak_extractor.compute(rchip, _cv_kpts)
        kpts = np.zeros((len(cv_kpts), 5), dtype=np.float32)
        desc = np.array(cv_descs, dtype=np.uint8)
        fx = 0
        for cv_kp in cv_kpts:
            (x,y) = cv_kp.pt
            theta = cv_kp.angle
            scale = float(cv_kp.size)**2 / 27
            detA  = 1./(scale)
            (a,c,d) = (detA, 0, detA)
            kpts[fx] = (x,y,a,c,d)
            fx += 1
        np.savez(chiprep_fpath, kpts, desc)
        return True
    ####
    
    # Compute SIFT Features
    print('Computing SIFT features')
    compute_sift_args = iter(zip(cx2_rchip_path, cx2_sift_path))
    _args = compute_sift_args.next()
    compute_sift_tasks = [(compute_sift, _args) for _args in compute_sift_args]
    parallelize_tasks(compute_sift_tasks, num_procs=8)

    # Compute FREAK Features
    print('Computing FREAK features')
    compute_freak_args = iter(zip(cx2_rchip_path, cx2_freak_path))
    compute_sift_tasks = [(compute_freak, _args) for _args in compute_freak_args]
    parallelize_tasks(compute_sift_tasks, num_procs=8)
