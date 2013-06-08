import os.path
from hotspotter.algo.imalgos import histeq
from hotspotter.algo.imalgos import adapt_histeq
from hotspotter.algo.imalgos import contrast_stretch
from hotspotter.tpl.other.shiftableBF import shiftableBF
from hotspotter.other.logger import logmsg, logdbg, logerr, logio, logwarn 
from PIL import Image 
import numpy as np
import subprocess


def get_task_list(hs, fpath_fn, work_fn, args_fn):
    # For each chip
    cx_list = []
    for cx in iter(hs.cm.get_valid_cxs()):
        # Test to see if the computation exists on disk
        if not os.path.exists(fpath_fn(hs, hs.cm.cx2_cid[cx])):
            cx_list += [cx]
    total = len(cx_list)
    task_list = [(work_fn, args_fn(hs, cx)) for cx in cx_list]
    return task_list
    

# --- COMPUTE CHIP --- 
def chip_fpath(hs, cid):
    return hs.iom.get_chip_fpath(cid)

def compute_chip(img_fpath, chip_fpath, roi, new_size, preproc_prefs):
    # Read image
    img = Image.open(img_fpath)
    [img_w, img_h] = [ gdim - 1 for gdim in img.size ]
    # Ensure ROI is within bounds
    [roi_x, roi_y, roi_w, roi_h] = [ max(0, cdim) for cdim in roi]
    roi_x2 = min(img_w, roi_x + roi_w)
    roi_y2 = min(img_h, roi_y + roi_h)
    # Crop out ROI: left, upper, right, lower
    raw_chip = img.crop((roi_x, roi_y, roi_x2, roi_y2))
    # Scale chip, but do not rotate
    pil_chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Preprocessing based on preferences
    if preproc_prefs.histeq_bit:
        pil_chip = histeq(pil_chip)
    if preproc_prefs.adapt_histeq_bit:
        pil_chip = Image.fromarray(adapt_histeq(np.asarray(pil_chip)))
    if preproc_prefs.contrast_stretch_bit:
        pil_chip = Image.fromarray(contrast_stretch(np.asarray(pil_chip)))
    if preproc_prefs.autocontrast_bit :
        pil_chip = ImageOps.autocontrast(pil_chip)
    if preproc_prefs.bilateral_filt_bit :
        pil_chip = shiftableBF(pil_chip)
    # Save chip to disk
    pil_chip.save(chip_fpath, 'PNG')

def compute_chip_args(hs, cx):
    # image info
    cm = hs.cm
    gm = hs.gm
    am = hs.am
    iom = hs.iom
    gx        = cm.cx2_gx[cx]
    img_fpath = gm.gx2_img_fpath(gx)
    # chip info
    cid = cm.cx2_cid[cx]
    roi = cm.cx2_roi[cx]
    chip_fpath    = iom.get_chip_fpath(cid)
    preproc_prefs = am.algo_prefs.preproc
    new_size      = cm._scaled_size(cx, dtype=int, rotated=False)
    return (img_fpath, chip_fpath, roi, new_size, preproc_prefs)

def compute_chip_driver(hs, cx, showmsg=True):
    cc_args = compute_chip_args(hs, cx)
    if showmsg:
        cid = hs.cm.cx2_cid[cx]
        chip_fpath = hs.iom.get_chip_fpath(cid)
        chip_fname = os.path.split(chip_fpath)[1]
        logmsg(('\nComputing Chip: cid=%d fname=%s\n'+hs.am.get_algo_name(['preproc'])) % (cid, chip_fname))
    compute_chip(*cc_args)


#compute_chip_driver(hs, 1)
#compute_chip_driver(hs, 2)
#compute_chip_driver(hs, 3)

# --- END COMPUTE CHIP ---


# --- COMPUTE FEATURES ---
def chiprep_fpath(hs, cid):
    return hs.iom.get_chiprep_fpath(cid)

def read_text_chiprep_file(outname):
    with open(outname, 'r') as file:
        # Read header
        ndims = int(file.readline())
        nfpts = int(file.readline())
        # Preallocate output
        fpts = np.zeros((nfpts, 5), dtype=np.float32)
        fdsc = np.zeros((nfpts, ndims), dtype=np.uint8)
        # iterate over lines
        lines = file.readlines()
        for kx, line in enumerate(lines):
            data = line.split(' ')
            fpts[kx,:] = np.array([np.float32(_)\
                                   for _ in data[0:5]], dtype=np.float32)
            fdsc[kx,:] = np.array([np.uint8(_)\
                                   for _ in data[5: ]], dtype=np.uint8)
        return (fpts, fdsc)
    
# TODO: orientation is currently a hack, should be computed by chips not chiprep
def compute_chiprep(chip_fpath, chiprep_fpath, exename, orientation):
    
    chip_unorient = Image.open(chip_fpath) 
    angle_degrees = orientation * 180 / np.pi
    chip_orient = chip_unorient.rotate(angle_degrees, resample=Image.BICUBIC, expand=1)
    orientchip_fpath = chip_fpath +'rotated.png'
    chip_orient.save(orientchip_fpath, 'PNG')
    outname = orientchip_fpath + '.hesaff.sift'
    args = '"' + orientchip_fpath + '"'
    cmd  = exename + ' ' + args
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if proc.returncode != 0:
        raise Exception('  * Failed to execute '+cmd+'\n  * OUTPUT: '+out)
    if not os.path.exists(outname):
        raise Exception('  * The output file doesnt exist: '+outname)
    fpts, fdsc = read_text_chiprep_file(outname)
    np.savez(chiprep_fpath, fpts, fdsc)
    return out

def compute_chiprep_args(hs, cx):
    cid           = hs.cm.cx2_cid[cx]
    chip_fpath    = hs.iom.get_chip_fpath(cid)
    chiprep_fpath = hs.iom.get_chiprep_fpath(cid)
    exename       = hs.iom.get_hesaff_exec()
    orientation   = hs.cm.cx2_theta[cx]
    return (chip_fpath, chiprep_fpath, exename, orientation)

def compute_chiprep_driver(hs, cx):
    chiprep_prefs = hs.am.algo_prefs.chiprep
    if chiprep_prefs.kpts_detector != 'heshesaff' or \
       chiprep_prefs.kpts_extractor != 'SIFT': 
        raise Exception('This computation is constrainted to only heshesaff')
    cf_args = compute_chiprep_args(hs, cx) 
    compute_chiprep(*cf_args)


