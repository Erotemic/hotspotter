
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

def get_cv2_params(cv2_class):
    from Pref import Pref
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

def compute_sift(rchip):
    return __compute(rchip, __detector, sift_extractor)
def compute_freak(rchip):
    return __compute(rchip, __detector, freak_extractor)
def precompute_sift(rchip_path, feats_path):
    return __precompute(rchip_path, feats_path, compute_sift)
def precompute_freak(rchip_path, feats_path):
    return __precompute(rchip_path, feats_path, compute_freak)

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

