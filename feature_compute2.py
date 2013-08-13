'''
Computes feature representations
'''
from __future__ import division
#from __init__ import *
# import hotspotter modules
import drawing_functions2 as df2
import algos
import params
import tpl.hesaff as hesaff
import helpers
from Parallelize import parallel_compute
from Pref import Pref
from Printable import DynStruct
# import scientific modules
from numpy import array, cos, float32, hstack, pi, round, sqrt, uint8, zeros
import numpy as np
import cv2
# import python modules
import sys
#print('LOAD_MODULE: feature_compute2.py')


def printDEBUG(msg):
    print msg

def default_feature_preferences():
    prefs = Pref()
    prefs.feat_type = Pref('HESAFF')
    prefs.whiten    = Pref(params.__WHITEN__)

#def old_cvkpts2_kpts(cvkpts)
    #kpts = zeros((len(cvkpts), 5), dtype=float32)
    #fx = 0
    #for cvkp in cvkpts:
        #(x,y) = cvkp.pt
        #theta = cvkp.angle
        #scale = float(cvkp.size)**2 / 27
        #detA  = 1./(scale)
        #(a,c,d) = (detA, 0, detA)
        #kpts[fx] = (x,y,a,c,d)
        #fx += 1
    #return (kpts, desc)

def root_sift(desc):
    ''' Takes the square root of each descriptor and returns the features in the
    range 0 to 255 (as uint8) '''
    desc_ = array([sqrt(d) for d in algos.norm_zero_one(desc)])
    desc_ = array([round(255.0 * d) for d in algos.norm_zero_one(desc_)], dtype=uint8)
    return desc_

def print_cvkpt(cvkp):
    import types
    from helpers import public_attributes
    if type(cvkp) == types.ListType:
        [print_cvkpt(kp) for kp in cvkp]
        return
    attr_list = public_attributes(cvkp)
    print('cv2 keypoint:')
    for attr in attr_list:
        exec('val = cvkp.'+attr)
        print('  '+attr+' '+str(val))
    
def cvkpts2_kpts(cvkpts):
    ''' Converts cv2 keypoints into the elliptical [x,y,a,d,c] format SIFT
    descriptors are computed with a radius of r = 3*sqrt(3*s);  s = (r/3)**2 / 3
    ;  s = r**2/27 '''
    kpts_xy      = array([cvkp.pt for cvkp in cvkpts])
    #kpts_octave = [cvkp.octave for cvkp in cv_fpts]
    #kpts_theta  = [cos(cvkp.angle) * tau / 360 for cvkp in cv_fpts] # tauday.com
    kpts_theta   = [cos(cvkp.angle) * pi / 180 for cvkp in cvkpts]
    kpts_radius  = [float(cvkp.size)/2.0 for cvkp in cvkpts]
    #kpts_ell    = [array((r, 0, r)) for r in kpts_radius]
    kpts_scale   = [1 / (r**2) for r in kpts_radius]
    kpts_ell     = [array((s, 0, s)) for s in kpts_scale]
    kpts = hstack((kpts_xy, kpts_ell))
    return kpts

def kpts_inv_sqrtm(kpts):
    kptsT = kpts.T
    xy = kptsT[0:2]
    a = kptsT[2]
    c = kptsT[3]
    d = kptsT[4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Manually Calculated sqrtm(inv(A) for A in kpts)
        aIS = 1/np.sqrt(a) 
        cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
        dIS = 1/np.sqrt(d)
    return np.hstack([xy, aIS, cIS, dIS]).T

def __compute(rchip, detector, extractor):
    'returns keypoints and descriptors'
    _cvkpts = detector.detect(rchip)  
    for cvkp in iter(_cvkpts): cvkp.angle = 0 # gravity vector
    cvkpts, cvdesc = extractor.compute(rchip, _cvkpts)
    kpts = cvkpts2_kpts(cvkpts)
    desc = array(cvdesc, dtype=uint8)
    return (kpts, desc)

def __precompute(rchip_path, feats_path, compute_func):
    'saves keypoints and descriptors to disk'
    rchip = cv2.imread(rchip_path)
    (kpts, desc) = compute_func(rchip)
    np.savez(feats_path, kpts, desc)
    return (kpts, desc)

# =======================================
# Global cv2 detectors and extractors      
# =======================================
def get_cv2_params(cv2_class):
    cv2_pref = Pref()
    for param_name in iter(cv2_class.getParams()):
        param_type = cv2_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            param_val = cv2_class.getInt(param_name)
        elif param_type == 1:
            param_val = cv2_class.getBool(param_name)
        elif param_type in [2,7, 8]:
            param_val = cv2_class.getDouble(param_name)
        else:
            raise Exception('Unknown cv2 param. name: '+str(param_name) + ' type: '+str(param_type))
        cv2_pref[param_name] = param_val
    return cv2_pref

def set_cv2_params(cv2_class, param_dict):
    for param_name, param_val in param_dict.iteritems():
        param_type = cv2_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            cv2_class.setInt(param_name, param_val)
        elif param_type == 1:
            cv2_class.setBool(param_name, param_val)
        elif param_type in [2,7]:
            cv2_class.setDouble(param_name, param_val)
        else:
            raise Exception('Unknown cv2 param. name: '+str(param_name) + ' type: '+str(param_type))
        cv2_pref[param_name] = param_val

cv2_detector_types  = ['BRISK', 'Dense', 'FAST', 'GFTT', 'HARRIS',
                       'MSER', 'ORB', 'SIFT', 'STAR', 'SURF', 'SimpleBlob']
cv2_extractor_types = ['BRISK', 'FAST', 'FREAK', 'GFTT', 'GridFAST',
                       'ORB', 'PyramidStar', 'SIFT', 'SURF']
# These extractors give segfaults
__off_list = ['FAST', 'GFTT', 'GridFAST', 'PyramidStar', 'ORB']
cv2_extractor_types = np.setdiff1d(cv2_extractor_types, __off_list).tolist()

#=========================================
# Create detector instances
#=========================================
#__detector  = cv2.FeatureDetector_create('SIFT')
__detector = cv2.FeatureDetector_create('SURF')
#__detector = cv2.GridAdaptedFeatureDetector(__detector)
__detector = cv2.PyramidAdaptedFeatureDetector(__detector)
#__detector = cv2.AdjusterAdapter(__detector)

#=========================================
# Create extractor instances
#=========================================
## SIFT extractor settings
sift_extractor = cv2.DescriptorExtractor_create('SIFT')
## SURF extractor settings
surf_extractor = cv2.DescriptorExtractor_create('SURF')
## FREAK extractor settings
freak_extractor = cv2.DescriptorExtractor_create('FREAK')
freak_extractor.setBool('orientationNormalized', False)

# =======================================
# Module Functions          
# =======================================

def compute_hesaff(rchip):
    return hesaff.compute_hesaff(rchip)
def compute_sift(rchip):
    return __compute(rchip, __detector, sift_extractor)
def compute_freak(rchip):
    return __compute(rchip, __detector, freak_extractor)

# =======================================
# Parallelizable Work Functions          
# =======================================

def precompute_hesaff(rchip_path, feats_path):
    return hesaff.precompute_hesaff(rchip_path, feats_path)
def precompute_sift(rchip_path, feats_path):
    return __precompute(rchip_path, feats_path, compute_sift)
def precompute_freak(rchip_path, feats_path):
    return __precompute(rchip_path, feats_path, compute_freak)

def load_features(feats_path):
    npz = np.load(feats_path)
    kpts = npz['arr_0']
    desc = npz['arr_1']
    return (kpts, desc)

#==========================================
# Dynamic Test Functions
#==========================================
type2_compute_func = {
    'HESAFF' : compute_hesaff,
    'SIFT'   : compute_sift,
    'FREAK'  : compute_freak }

type2_precompute_func = {
    'HESAFF' : precompute_hesaff,
    'SIFT'   : precompute_sift,
    'FREAK'  : precompute_freak }

def compute_features(rchip, extractor_type):
    return type2_compute_func[extractor_type](rchip)

def cv2_kpts(rchip, detect_type, pyramid=False, grid=False):
    detector  = cv2.FeatureDetector_create(detect_type)
    if pyramid: 
        detector = cv2.PyramidAdaptedFeatureDetector(detector)
    if grid:
        detector = cv2.GridAdaptedFeatureDetector(detector)
    cvkpts = detector.detect(rchip)  
    return cvkpts

def cv2_feats(rchip, extract_type, detect_type, pyramid=False, grid=False):
    detector  = cv2.FeatureDetector_create(detect_type)
    extractor = cv2.DescriptorExtractor_create(extract_type)
    if grid:
        detector = cv2.GridAdapatedFeatureDetector(detector)
    if pyramid: 
        detector = cv2.PyramidAdaptedFeatureDetector(detector)
    (kpts, desc) =  __compute(rchip, detector, extractor)
    return (kpts, desc)

# =======================================
# Main Script 
# =======================================

class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.is_binary = False
        self.cx2_desc = None
        self.cx2_kpts = None
        self.feat_type = None
        # DEP
        self.cx2_feats_hesaff = []
        self.cx2_feats_sift   = []
        self.cx2_feats_freak  = []

    
def load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, feat_type, feat_uid, cache_dir):
    print('Loading '+feat_type+' features: UID='+str(feat_uid))
    cx2_kpts_fpath = cache_dir + '/cx2_kpts'+feat_uid+'.npz'
    cx2_desc_fpath = cache_dir + '/cx2_desc'+feat_uid+'.npz'
    # Try to read cache
    cx2_kpts = helpers.tryload(cx2_kpts_fpath)
    cx2_desc = helpers.tryload(cx2_desc_fpath)
    if (not cx2_kpts is None and not cx2_desc is None):
        # This is pretty dumb. Gotta have a more intelligent save/load
        cx2_desc_ = cx2_desc.tolist()
        cx2_kpts  = cx2_kpts.tolist()
        print(' * Loaded cx2_kpts and cx2_desc from cache')
        #print all([np.all(desc == desc_) for desc, desc_ in zip(cx2_desc, cx2_desc_)])
    else:
        # Recompute if you cant
        print(' * Loading original '+feat_type+' features')
        cx2_feat_path = [ feat_dir+'/CID_%d_%s.npz' % (cid, feat_type) for cid in cx2_cid]
        # Compute features
        precompute_func = type2_precompute_func[feat_type]
        parallel_compute(precompute_func, [cx2_rchip_path, cx2_feat_path], params.__NUM_PROCS__)

        # Load features
        cx2_feats = parallel_compute(load_features, [cx2_feat_path], 1)
        cx2_kpts  = [k for (k,d) in cx2_feats]
        cx2_desc  = np.array([d for (k,d) in cx2_feats])
        # Whiten descriptors
        if params.__WHITEN_FEATS__:
            print(' * Whitening features')
            #print (' * Stacking '+str(len(cx2_desc))+' features')
            #print helpers.info(cx2_desc, 'cx2_desc')
            ax2_desc = np.vstack(cx2_desc)
            ax2_desc_white = algos.scale_to_byte(algos.whiten(ax2_desc))
            #print (' * '+helpers.info(ax2_desc, 'ax2_desc'))
            #print (' * '+helpers.info(ax2_desc_white, 'ax2_desc_white'))
            index = 0
            offset = 0
            #print ('Looping through '+str(len(cx2_desc))+' features')
            for cx in xrange(len(cx2_desc)):
                old_desc = cx2_desc[cx]
                print (' * '+helpers.info(old_desc, 'old_desc'))
                offset = len(old_desc)
                new_desc = ax2_desc_white[index:(index+offset)]
                #print ('index=%r ; offset=%r ; new_desc.shape=%r' % (offset, index, helpers.info(new_desc,'new_desc')))
                cx2_desc[cx] = new_desc
                index += offset
            #print ('index=%r ; offset=%r ; new_desc.shape=%r' % (index, offset, new_desc.shape,))
            #print ' * '+helpers.info(cx2_desc, 'cx2_desc')
        else:
            print(' * not whitening features')
        helpers.save_npz(cx2_desc_fpath, cx2_desc)
        helpers.save_npz(cx2_kpts_fpath, cx2_kpts)
    #if __WHITEN_INDIVIDUAL__: # I dont think this is the way to go
    #    cx2_desc = [algos.whiten(d) for (k,d) in cx2_feats]

    # cache the data
    return cx2_kpts, cx2_desc
    
def load_chip_features(hs_dirs, hs_tables, hs_cpaths):
    print('=============================')
    print('Computing and loading features')
    print('=============================')
    # --- GET INPUT --- #
    # Paths to features
    feat_dir       = hs_dirs.feat_dir
    cache_dir      = hs_dirs.cache_dir
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    cx2_cid        = hs_tables.cx2_cid
    hs_feats = HotspotterChipFeatures()
    # Load all the types of features
    feat_uid = params.get_feat_uid()
    cx2_kpts, cx2_desc = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, params.__FEAT_TYPE__, feat_uid, cache_dir)
    hs_feats.feat_type = params.__FEAT_TYPE__
    hs_feats.cx2_kpts = cx2_kpts
    hs_feats.cx2_desc = cx2_desc
    #hs_feats.cx2_feats_sift   = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'SIFT')
    #hs_feats.cx2_feats_freak  = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'FREAK')
    print('=============================')
    print('Done computing and loading features')
    print('=============================\n')
    return hs_feats

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('IN: feature_compute2.py: __name__ == \'__main__\'')

    __DEV_MODE__ = True
    if __DEV_MODE__ or 'test' in sys.argv:
        import load_data2
        import match_chips2 as mc2
        import chip_compute2
        # --- CHOOSE DATABASE --- #
        db_dir = load_data2.DEFAULT
        hs = load_data2.HotSpotter(db_dir, load_matcher=False)
        cx2_desc = hs.feats.cx2_desc
        cx2_kpts = hs.feats.cx2_kpts
        cx2_cid  = hs.tables.cx2_cid
        cx2_nx   = hs.tables.cx2_nx
        nx2_name = hs.tables.nx2_name

    exec(df2.present())
