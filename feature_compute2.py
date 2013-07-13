from __future__ import division
from hotspotter.Parallelize import parallel_compute
from hotspotter.helpers import normalize
from hotspotter.other.ConcretePrintable import DynStruct
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

def __comp_cv_feats(rchip_path, feats_path, detector, extractor):
    rchip = cv2.imread(rchip_path)
    _cvkpts = detector.detect(rchip)  
    # gravity vector
    for cvkp in iter(_cvkpts): cvkp.angle = 0
    cvkpts, cvdesc = extractor.compute(rchip, _cvkpts)
    kpts = cvkpts2_kpts(cvkpts)
    desc = array(cvdesc, dtype=uint8)
    np.savez(feats_path, kpts, desc)
    return (kpts, desc)

# =======================================
# Global opencv detectors and extractors      
# =======================================
## Common keypoint detector
#common_detector  = cv2.FeatureDetector_create('SURF')
common_detector  = cv2.FeatureDetector_create('SIFT')
## SIFT extractor settings
sift_extractor = cv2.DescriptorExtractor_create('SIFT')
## FREAK extractor settings
freak_extractor = cv2.DescriptorExtractor_create('FREAK')
freak_extractor.setBool('orientationNormalized', False)

# =======================================
# Parallelizable Work Functions          
# =======================================

def compute_hesaff(rchip_path, feats_path):
    return hotspotter.tpl.hesaff.compute_hesaff(rchip_path, feats_path)

def compute_sift(rchip_path, feats_path):
    return __comp_cv_feats(rchip_path, feats_path,
                           common_detector, sift_extractor)

def compute_freak(rchip_path, feats_path):
    return __comp_cv_feats(rchip_path, feats_path,
                           common_detector, freak_extractor)

def load_features(feats_path):
    npz = np.load(feats_path)
    kpts = npz['arr_0']
    desc = npz['arr_1']
    return (kpts, desc)

# =======================================
# Main Script 
# =======================================

class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.cx2_hesaff_feats = []
        self.cx2_sift_feats   = []
        self.cx2_freak_feats  = []

def load_chip_features(hs_dirs, hs_tables, hs_cpaths):
    print('\n=============================')
    print('Computing and loading features')
    print('=============================')

    # --- BUILD TASK INFORMATION --- #
    # Paths to features
    feat_dir       = hs_dirs.feat_dir
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    cx2_cid        = hs_tables.cx2_cid
    
    cx2_hesaff_path = [ feat_dir+'/CID_%d_hesaff.npz' % cid for cid in cx2_cid]
    cx2_sift_path   = [ feat_dir+'/CID_%d_sift.npz'   % cid for cid in cx2_cid]
    cx2_freak_path  = [ feat_dir+'/CID_%d_freak.npz'  % cid for cid in cx2_cid]
    
    # --- COMPUTE FEATURES --- # 
    print('Computing features')
    parallel_compute(compute_hesaff, [cx2_rchip_path, cx2_hesaff_path])
    parallel_compute(compute_sift,   [cx2_rchip_path, cx2_sift_path])
    #parallel_compute(compute_freak,  [cx2_rchip_path, cx2_freak_path]) 
    # --- LOAD FEATURES --- # 
    print('Loading features')
    cx2_hesaff_feats = parallel_compute(load_features, [cx2_hesaff_path], 1)
    cx2_sift_feats   = parallel_compute(load_features, [cx2_sift_path], 1)
    #cx2_freak_feats   = parallel_compute(load_features, [cx2_freak_path])

    hs_feats = HotspotterChipFeatures()
    hs_feats.cx2_hesaff_feats = cx2_hesaff_feats
    hs_feats.cx2_sift_feats   = cx2_sift_feats
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

# GRAVEYARD
    #cx2_sift_kpts, cx2_sift_desc = \
    #     ([ k for k,d in cx2_sift_feats ], [ d for k,d in cx2_sift_feats ])

'''
from feature_compute2 import *
detector = common_detector
extractor = sift_extractor

rchip = cv2.imread(rchip_path)
_cvkpts = detector.detect(rchip)  
print_cvkpt(_cvkpts)
'''    
