from __future__ import division
from hotspotter.Parallelize import parallel_compute
from hotspotter.helpers import normalize
from hotspotter.other.ConcretePrintable import DynStruct, Pref
from numpy import array, cos, float32, hstack, pi, round, sqrt, uint8, zeros
import hotspotter.tpl.cv2 as cv2
import hotspotter.tpl.hesaff
import numpy as np

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
    '''
    Takes the square root of each descriptor and
    returns the features in the range 0 to 255 (as uint8) '''
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
    print('opencv keypoint:')
    for attr in attr_list:
        exec('val = cvkp.'+attr)
        print('  '+attr+' '+str(val))
    
def cvkpts2_kpts(cvkpts):
    ''' 
    Converts opencv keypoints into the elliptical [x,y,a,d,c] format 
    SIFT descriptors are computed with a radius of 
    r = 3*sqrt(3*s);  s = (r/3)**2 / 3 ;  s = r**2/27 '''
    kpts_xy      = array([cvkp.pt for cvkp in cvkpts])
    kpts_radius  = [float(cvkp.size)/2.0 for cvkp in cvkpts]
    kpts_scale   = [1 / (r**2) for r in kpts_radius]
    #kpts_octave = [cvkp.octave for cvkp in cv_fpts]
    #kpts_theta  = [cos(cvkp.angle) * tau / 360 for cvkp in cv_fpts] # tauday.com
    kpts_theta   = [cos(cvkp.angle) * pi / 180 for cvkp in cvkpts]
    kpts_ell     = [array((s, 0, s)) for s in kpts_scale]
    kpts = hstack((kpts_xy, kpts_ell))
    return kpts

def __compute(rchip, detector, extractor):
    _cvkpts = detector.detect(rchip)  
    # gravity vector
    for cvkp in iter(_cvkpts): cvkp.angle = 0
    cvkpts, cvdesc = extractor.compute(rchip, _cvkpts)
    kpts = cvkpts2_kpts(cvkpts)
    desc = array(cvdesc, dtype=uint8)
    return (kpts, desc)

def __precompute(rchip_path, feats_path, compute_fn):
    rchip = cv2.imread(rchip_path)
    (kpts, desc) = compute_fn(rchip)
    np.savez(feats_path, kpts, desc)
    return (kpts, desc)


# =======================================
# Global opencv detectors and extractors      
# =======================================
## Common keypoint detector
def get_opencv_params(opencv_class):
    opencv_pref = Pref()
    for param_name in opencv_class.getParams():
        param_type = opencv_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            param_val = opencv_class.getInt(param_name)
        elif param_type == 1:
            param_val = opencv_class.getBool(param_name)
        elif param_type in [2,7]:
            param_val = opencv_class.getDouble(param_name)
        else:
            raise Exception('Unknown opencv param. name: '+str(param_name) + ' type: '+str(param_type))
        opencv_pref[param_name] = param_val
        return opencv_pref

def set_opencv_params(opencv_class, param_dict):
    for param_name, param_val in param_dict.iteritems():
        param_type = opencv_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            opencv_class.setInt(param_name, param_val)
        elif param_type == 1:
            opencv_class.setBool(param_name, param_val)
        elif param_type in [2,7]:
            opencv_class.setDouble(param_name, param_val)
        else:
            raise Exception('Unknown opencv param. name: '+str(param_name) + ' type: '+str(param_type))
        opencv_pref[param_name] = param_val

#http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=surffeaturedetector#surffeaturedetector
'''
"FAST"  FastFeatureDetector
"STAR"  StarFeatureDetector
"SIFT"  SIFT (nonfree module) - Hessian Trace
"SURF"  SURF (nonfree module) - Hessian Determinant
"ORB"  ORB
"BRISK"  BRISK
"MSER"  MSER
"GFTT" GoodFeaturesToTrackDetector
"HARRIS"  GoodFeaturesToTrackDetector with Harris detector enabled
"Dense"  DenseFeatureDetector
"SimpleBlob"  SimpleBlobDetector
'''
detector_options = ['SIFT', 'SURF', 'MSER', 'STAR', 'DENSE', 'HARRIS', 'SimpleBlob'] 

extractor_options = ['SIFT', 'SURF', 'FREAK', 'ORB', 'BRISK', 'FAST', 'GFTT',
                     'GridFAST', 'PyramidStar']

__detector = cv2.FeatureDetector_create('MSER')
detector_params = get_opencv_params(__detector)
detector_params.printme()


#__detector  = cv2.FeatureDetector_create('SIFT')
## SIFT extractor settings
sift_extractor = cv2.DescriptorExtractor_create('SIFT')
## SURF extractor settings
surf_extractor = cv2.DescriptorExtractor_create('SURF')
## FREAK extractor settings
freak_extractor = cv2.DescriptorExtractor_create('FREAK')
freak_extractor.setBool('orientationNormalized', False)
# Adapt the descriptor
__detector = cv2.PyramidAdaptedFeatureDetector(__detector)
#__detector = cv2.GridAdapatedFeatureDetector(__detector)
#__detector = cv2.AdjusterAdapter(__detector)

# =======================================
# Module Functions          
# =======================================

def compute_hesaff(rchip):
    return hotspotter.tpl.hesaff.compute_hesaff(rchip)
def compute_sift(rchip):
    return __compute(rchip, __detector, sift_extractor)
def compute_freak(rchip):
    return __compute(rchip, __detector, freak_extractor)

type2_compute_fn = {
    'HESAFF' : compute_hesaff,
    'SIFT'   : compute_sift,
    'FREAK'  : compute_freak
}
def compute_features(rchip, type):
    return type2_compute_fn[type](rchip)

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
    cx2_hesaff_path = [ feat_dir+'/CID_%d_hesaff.npz' % cid for cid in cx2_cid]
    cx2_sift_path   = [ feat_dir+'/CID_%d_sift.npz'   % cid for cid in cx2_cid]
    cx2_freak_path  = [ feat_dir+'/CID_%d_freak.npz'  % cid for cid in cx2_cid]
    # --- COMPUTE FEATURES --- # 
    print('Computing features')
    parallel_compute(precompute_hesaff, [cx2_rchip_path, cx2_hesaff_path])
    parallel_compute(precompute_sift,   [cx2_rchip_path, cx2_sift_path])
    parallel_compute(precompute_freak,  [cx2_rchip_path, cx2_freak_path]) 
    # --- LOAD FEATURES --- # 
    print('Loading features')
    cx2_feats_hesaff = parallel_compute(load_features, [cx2_hesaff_path], 1)
    cx2_feats_sift   = parallel_compute(load_features, [cx2_sift_path], 1)
    cx2_feats_freak  = parallel_compute(load_features, [cx2_freak_path], 1)
    # --- BUILD OUTPUT --- #
    hs_feats = HotspotterChipFeatures()
    hs_feats.cx2_feats_hesaff = cx2_feats_hesaff
    hs_feats.cx2_feats_sift   = cx2_feats_sift
    hs_feats.cx2_feats_freak  = cx2_feats_freak
    print('=============================')
    print('Done computing and loading features')
    print('=============================\n\n')
    return hs_feats

if __name__ == '__main__':
    import load_data2
    import chip_compute2
    from multiprocessing import freeze_support
    freeze_support()
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.MOTHERS
    # --- LOAD DATA --- #
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # --- LOAD CHIPS --- #
    hs_cpaths = chip_compute2.load_chip_paths(hs_dirs, hs_tables)
    # --- LOAD FEATURES --- #
    hs_feats  = load_chip_features(hs_dirs, hs_tables, hs_cpaths)
