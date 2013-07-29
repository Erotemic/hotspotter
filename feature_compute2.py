from __future__ import division, print_function
import drawing_functions2 as df2
from hotspotter.Parallelize import parallel_compute
from hotspotter.helpers import normalize
from hotspotter.other.ConcretePrintable import DynStruct, Pref
from numpy import array, cos, float32, hstack, pi, round, sqrt, uint8, zeros
import sys
import hotspotter.tpl.hesaff
import numpy as np
import load_data2, chip_compute2
import cv2


__NUM_PROCS__ = 9
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
    desc_ = array([sqrt(d) for d in normalize(desc)])
    desc_ = array([round(255.0 * d) for d in normalize(desc_)], dtype=uint8)
    return desc_

def print_cvkpt(cvkp):
    import types
    from hotspotter.helpers import public_attributes
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
    #kpts_ell     = [array((r, 0, r)) for r in kpts_radius]
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
    return hotspotter.tpl.hesaff.compute_hesaff(rchip)
def compute_sift(rchip):
    return __compute(rchip, __detector, sift_extractor)
def compute_freak(rchip):
    return __compute(rchip, __detector, freak_extractor)

# =======================================
# Parallelizable Work Functions          
# =======================================

def precompute_hesaff(rchip_path, feats_path):
    return hotspotter.tpl.hesaff.precompute_hesaff(rchip_path, feats_path)
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
    #
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.cx2_feats_hesaff = []
        self.cx2_feats_sift   = []
        self.cx2_feats_freak  = []
        self.cx2_desc = None
        self.cx2_kpts = None
        self.feat_type = None
    #
    def set_feat_type(self, feat_type):
        if feat_type == self.feat_type:
            print('Feature type is already: '+feat_type)
            return
        print('Setting feature type to: '+feat_type)
        self.feat_type = feat_type
        if self.feat_type == 'HESAFF':
            cx2_feats = self.cx2_feats_hesaff
        elif self.feat_type == 'SIFT':
            cx2_feats = self.cx2_feats_sift
        elif self.feat_type == 'FREAK':
            cx2_feats = self.cx2_feats_freak
        self.cx2_desc  = [d for (k,d) in cx2_feats]
        self.cx2_kpts  = [k for (k,d) in cx2_feats]


def load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, feat_type):
    cx2_feat_path = [ feat_dir+'/CID_%d_%s.npz' % (cid, feat_type) for cid in cx2_cid]
    # Compute features
    print('Loading '+feat_type)
    precompute_func = type2_precompute_func[feat_type]
    parallel_compute(precompute_func, [cx2_rchip_path, cx2_feat_path], __NUM_PROCS__)
    # Load features
    cx2_feats = parallel_compute(load_features, [cx2_feat_path], 1)
    return cx2_feats
    
    

def load_chip_features(hs_dirs, hs_tables, hs_cpaths):
    print('\n=============================')
    print('Computing and loading features')
    print('=============================')
    # --- GET INPUT --- #
    # Paths to features
    feat_dir       = hs_dirs.feat_dir
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    cx2_cid        = hs_tables.cx2_cid
    # --- COMPUTE FEATURE PATHS --- # 
    #cx2_hesaff_path = [ feat_dir+'/CID_%d_hesaff.npz' % cid for cid in cx2_cid]
    #cx2_sift_path   = [ feat_dir+'/CID_%d_sift.npz'   % cid for cid in cx2_cid]
    #cx2_freak_path  = [ feat_dir+'/CID_%d_freak.npz'  % cid for cid in cx2_cid]
    ## --- COMPUTE FEATURES --- # 
    #print('Computing features')
    #parallel_compute(precompute_hesaff, [cx2_rchip_path, cx2_hesaff_path], __NUM_PROCS__)
    #parallel_compute(precompute_sift,   [cx2_rchip_path, cx2_sift_path],   __NUM_PROCS__)
    #parallel_compute(precompute_freak,  [cx2_rchip_path, cx2_freak_path],  __NUM_PROCS__) 
    ## --- LOAD FEATURES --- # 
    #print('Loading features')
    #cx2_feats_hesaff = parallel_compute(load_features, [cx2_hesaff_path], 1)
    #cx2_feats_sift   = parallel_compute(load_features, [cx2_sift_path], 1)
    #cx2_feats_freak  = parallel_compute(load_features, [cx2_freak_path], 1)
    # --- BUILD OUTPUT --- #
    hs_feats = HotspotterChipFeatures()
    #hs_feats.cx2_feats_hesaff = cx2_feats_hesaff
    #hs_feats.cx2_feats_sift   = cx2_feats_sift
    #hs_feats.cx2_feats_freak  = cx2_feats_freak
    hs_feats.cx2_feats_hesaff = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'HESAFF')
    hs_feats.cx2_feats_sift   = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'SIFT')
    hs_feats.cx2_feats_freak  = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'FREAK')
    print('=============================')
    print('Done computing and loading features')
    print('=============================\n\n')
    return hs_feats

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    __DEV_MODE__ = True
    if __DEV_MODE__ or 'devmode' in sys.argv:
        import load_data2
        import chip_compute2
        # --- CHOOSE DATABASE --- #
        db_dir = load_data2.MOTHERS
        # --- LOAD DATA --- #
        hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
        # --- LOAD CHIPS --- #
        hs_cpaths = chip_compute2.load_chip_paths(hs_dirs, hs_tables)
        # --- LOAD FEATURES --- #
        hs_feats  = load_chip_features(hs_dirs, hs_tables, hs_cpaths)
