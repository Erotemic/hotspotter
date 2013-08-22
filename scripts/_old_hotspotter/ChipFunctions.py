from __future__ import print_function, division
from hotspotter.algo.imalgos import adapt_histeq
from hotspotter.algo.imalgos import contrast_stretch
from hotspotter.algo.imalgos import histeq
from hotspotter.other.logger import logmsg, logdbg, logerr, logio, logwarn, func_log
#from hotspotter.tpl.other.shiftableBF import shiftableBF
import cv2
import hotspotter.Parallelize
import os.path
import types

from PIL import Image 
import numpy as np
import subprocess

# Override the print function to logmsg
def print(*args):
    if len(args) == 1:
        logmsg(str(args[0]))
    else:
        logmsg(' '.join(args))

@func_log
def __cx_task_list(hs, cx_list, cx_fpath_fn, cx_compute_fn, cx_args_fn, force_recompute):
    # For each chip
    uncomp_cx_list = []
    for cx in iter(cx_list):
        # Test to see if the computation exists on disk
        if not os.path.exists(cx_fpath_fn(hs, hs.cm.cx2_cid[cx])) or force_recompute:
            uncomp_cx_list.append(cx)
    task_list = [(cx_compute_fn, cx_args_fn(hs, cx)) for cx in uncomp_cx_list]
    return task_list

@func_log
def __cx_precompute(hs, cx_list, num_procs, cx_fpath_fn, cx_compute_fn, cx_args_fn, force_recompute=False):
    if cx_list is None: 
        cx_list =  hs.cm.get_valid_cxs()
    print('  * Requested %d chips' % len(cx_list))
    task_list = __cx_task_list(hs, cx_list, cx_fpath_fn, cx_compute_fn, cx_args_fn, force_recompute)
    if len(task_list) == 0:
        print('  * The chips are all clean')
        return True
    print('  * There are %d uncomputed tasks' % len(task_list))
    num_procs = hs.core_prefs.num_procs if num_procs is None else num_procs
    hotspotter.Parallelize.parallelize_tasks(task_list, num_procs)
    return True

@func_log
def precompute_chips(hs, cx_list=None, num_procs=None, force_recompute=False,
                     showmsg=True):
    print('Ensuring chips are computed')
    if showmsg and not cx_list is None:
        print('  * '+hs.am.get_algo_name(['preproc']))
        for cx in cx_list: 
            cid = hs.cm.cx2_cid[cx]
            chip_fpath = hs.iom.get_chip_fpath(cid)
            chip_fname = os.path.split(chip_fpath)[1]
            print(('    * Ensuring Chip: cid=%d fname=%s') % (cid, chip_fname))
    return __cx_precompute(hs,
                           cx_list,
                           num_procs,
                           chip_fpath_fn,
                           compute_chip,
                           compute_chip_args,
                           force_recompute)
@func_log
def precompute_chipreps(hs, cx_list=None, num_procs=None, force_recompute=False):
    print('\nEnsuring chip representations are computed')
    precompute_chips(hs, cx_list, num_procs, force_recompute=False, showmsg=False)
    return __cx_precompute(hs,
                           cx_list,
                           num_procs,
                           chiprep_fpath_fn,
                           compute_chiprep, 
                           compute_chiprep_args,
                           force_recompute)

# --- COMPUTE CHIP --- 
def chip_fpath_fn(hs, cid):
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
        raise NotImplemented('Bilateral filter implementation removed.')
        #pil_chip = shiftableBF(pil_chip)
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

#compute_chip_driver(hs, 1)
#compute_chip_driver(hs, 2)
#compute_chip_driver(hs, 3)

# --- END COMPUTE CHIP ---

# --- COMPUTE FEATURES ---
def chiprep_fpath_fn(hs, cid):
    return hs.iom.get_chiprep_fpath(cid)

def read_text_chiprep_file(outname):
    'Reads output from external keypoint detectors like hesaff'
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
def compute_chiprep_external(chip_fpath, chiprep_fpath, exename, orientation):
    'Runs external keypoint detetectors like hesaff'
    chip_orient = read_oriented_chip(chip_fpath, orientation)
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
    return fpts, fdsc

def read_oriented_chip(chip_fpath, orientation):
    'Reads and rotates a chip'
    chip_unorient = Image.open(chip_fpath) 
    chip_orient = chip_unorient.rotate\
            (orientation*180/np.pi, resample=Image.BICUBIC, expand=1)
    return chip_orient

def normalize(array, dim=0):
    'normalizes a numpy array from 0 to 1'
    array_max = array.max(dim)
    array_min = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    return np.divide(np.subtract(array, array_min), array_exnt)

def compute_chiprep(chip_fpath, chiprep_fpath, detector, extractor, orientation, gravity, params_dict):
    'A workfunction which computes keypoints and descriptors'
    if detector == 'heshesaff':
        return compute_chiprep_external(chip_fpath, chiprep_fpath, extractor, orientation)
    # Read image and convert to grayscale
    #img_ = cv2.imread(chip_fpath)
    #img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = np.asarray(read_oriented_chip(chip_fpath, orientation))
    # Create feature detectors / extractors
    fpts_detector_ = cv2.FeatureDetector_create(detector)

    # Set detector parameters
    for key, val in params_dict.iteritems():
        if type(val) in [types.BooleanType]:
            fpts_detector_.setBool(key, val)
        if type(val) in [types.IntType]:
            fpts_detector_.setInt(key, val)
        elif type(val) in [types.FloatType]:
            fpts_detector_.setDouble(key, val)

    #fpts_detector  = cv2.GridAdaptedFeatureDetector(fpts_detector_)
    fpts_detector = fpts_detector_
    fdcs_extractor = cv2.DescriptorExtractor_create(extractor)
    # Get cv_fpts and cv_fdsc 
    cv_fpts1 = fpts_detector.detect(img)  
    if gravity:
        for cv_kp in cv_fpts1:
            cv_kp.angle = 0
    (cv_fpts, cv_fdsc1) = fdcs_extractor.compute(img, cv_fpts1)
    # Root SIFT normalized between 0-1
    cv_fdsc2 = normalize(np.array(
        [np.sqrt(dsc) for dsc in normalize(cv_fdsc1)]  ))
    # Scale up to 255 uint8
    fdsc = np.array([np.round(255.0*dsc) for dsc in cv_fdsc2], dtype=np.uint8)
    
    # Get fpts
    xy_list = np.array([cv_kp.pt for cv_kp in cv_fpts])
    #angle_list = [[np.cos(cv_kp.angle)*tau/360] for cv_kp in cv_fpts]  # tauday.com
    #angle_list = [np.cos(cv_kp.angle)*np.pi/180 for cv_kp in cv_fpts]
    #octave_list = [cv_kp.octave for cv_kp in cv_fpts]
    # SIFT descriptors are computed with a radius of 
    # r = 3*np.sqrt(3*s)
    # s = (r/3)**2 / 3 
    # s = r**2/27
    radius_list = [float(cv_kp.size)/2.0 for cv_kp in cv_fpts]
    scale_list  = [1/(r**2) for r in radius_list]

    ell_list = [np.array((s, 0, s)) for s in scale_list]
    fpts_ = np.hstack((xy_list, ell_list))
    fpts = np.array(fpts_, dtype=np.float32)
    np.savez(chiprep_fpath, fpts, fdsc)

    '''
    toShow = cv2.drawKeypoints(img, cv_fpts1)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    toShow = cv2.drawKeypoints(img, cv_fpts, flags=flags)
    fig = plt.figure(0)
    fig.clf()
    ax  = plt.subplot(111)
    ax.imshow(toShow)
    fig.show()
    fig.canvas.draw()
    '''
    return fpts, fdsc

def compute_chiprep_args(hs, cx): 
    # TODO: Split into global and local arguments. 
    # Global arguments can be put into a list (or dict), and that pointer can be passed
    # around. Local arguments can be passed as normal
    chiprep_prefs = hs.am.algo_prefs.chiprep
    cid           = hs.cm.cx2_cid[cx]
    chip_fpath    = hs.iom.get_chip_fpath(cid)
    chiprep_fpath = hs.iom.get_chiprep_fpath(cid)
    detector      = chiprep_prefs.kpts_detector
    extractor     = chiprep_prefs.kpts_extractor
    gravity       = chiprep_prefs.use_gravity_vector
    if detector == 'heshesaff' and extractor == 'SIFT':
        extractor   = hs.iom.get_hesaff_exec()
        params_dict = {}
    else:
        params_dict = chiprep_prefs[detector+'_params'].to_dict()
    orientation   = hs.cm.cx2_theta[cx]
    return (chip_fpath, chiprep_fpath, detector, extractor, orientation,
            gravity, params_dict)
