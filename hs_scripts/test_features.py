from __future__ import division, print_function
import drawing_functions2 as df2
import feature_compute2 as fc2
import itertools
import sys

# IDEAS: 
#  put a descriptor at a spot.
#  see how it varies as you slowly move it away
# ----
# learn what that should be such that is linear with NN? 
# ----
# compute all detectors and just visualize those
# ----
# make sure that keypoints are plotted correctly w/ cv2 keypoints
# ----
# implement coverage measure

# =======================================
# Tests
# =======================================

def print(msg):
    sys.stdout.write(msg+'\n')
    sys.stdout.flush()

def test_img1():
    return df2.test_img(4)

def test_img2():
    return df2.test_img(1)

def print_detector_params():
    for detector_type in iter(fc2.cv2_detector_types):
        print('Printing Params cv2.FeatureDetector: %r ' % detector_type)
        detector = cv2.FeatureDetector_create(detector_type)
        detector_params = get_cv2_params(detector)
        print(str(detector_params).replace('Pref\n',''))

def show_all_detector_types(fignum=0, pyramid=True, grid=True, compare=False):
    print('Detecting features of all types')
    if compare: 
        show_all_detector_types(fignum=1.221, pyramid=False, grid=False)
        show_all_detector_types(fignum=1.222, pyramid=False, grid=True)
        show_all_detector_types(fignum=1.223, pyramid=True,  grid=False)
        show_all_detector_types(fignum=1.224, pyramid=True,  grid=True)
        return
    test_img = test_img1()
    for xx, detector_type in enumerate(fc2.cv2_detector_types):
        detector_args = ['','+grid'][grid]+['','+pyramid'][pyramid]
        print('Testing detector=%s' % (detector_type+detector_args))
        try: 
            cvkpts = fc2.cv2_kpts(test_img, detector_type, pyramid=pyramid, grid=grid)
            _img = df2.cv2_draw_kpts(test_img, cvkpts)
            df2.imshow(_img,
                       fignum=fignum+xx, 
                       title='%s #kpts=%d ' % (detector_args, len(cvkpts)), 
                       figtitle=detector_type)
        except Exception as ex:
            print(repr(ex))

def compute_all_desc_extrac_permutations():
    print('Computing all descriptor / extractor permutations')
    test_img = test_img1()
    # Try all combinations of feature detectors / extractors
    feat_type_perms = itertools.product(fc2.cv2_detector_types, fc2.cv2_extractor_types)
    for detector_type, extract_type in feat_type_perms:
        print('Testing detector+extractor=%s+%s' % (detector_type, extract_type))
        kpts, desc = fc2.cv2_feats(test_img, extract_type, detector_type)
        print(' * (descriptor --- extractor) array shape: %r --- %r ' % (kpts.shape, desc.shape))

if __name__ == '__main__':
    show_all_detector_types(fignum=0, compare=True)
    #compute_all_desc_extrac_permutations()
    df2.present()
