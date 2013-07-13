from __future__ import division
import hotspotter.tpl.cv2 as cv2
import hotspotter.tpl.hesaff
from hotspotter.helpers import normalize
from numpy import array, cos, float32, hstack, pi, round, sqrt, uint8, zeros
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
