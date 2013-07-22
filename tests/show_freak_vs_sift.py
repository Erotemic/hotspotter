import cv2
from pylab import *
from hotspotter.helpers import Timer, figure
import os
figure(1, doclf=True, title='SIFT Descriptors')
figure(2, doclf=True, title='FREAK Descriptors')

# Define Image
chip_fpath = 'D:/data/work/PZ_flankhack/images/img-0000001.jpg'
# Read Image
img_ = cv2.imread(chip_fpath)
img  = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
imshow(img)
set_cmap('gray')
params = { 'kpts_type' :'SIFT'}
# Detect and Extract
print('Detect and Extract')
kpts_detector3 = cv2.FeatureDetector_create(params['kpts_type'])
kpts_detector = cv2.PyramidAdaptedFeatureDetector(kpts_detector3, maxLevel=5)
#kpts_detector  = cv2.GridAdaptedFeatureDetector(kpts_detector3) # max number of features
with Timer(name=params['kpts_type']+' Keypoint Detector'):
    cv_kpts2 = kpts_detector.detect(img)  
sift_desc_extractor = cv2.DescriptorExtractor_create('SIFT')
freak_desc_extractor = cv2.DescriptorExtractor_create('FREAK')
with Timer(name='SIFT Descriptors'):
    (sift_cv_kpts, sift_cv_desc)   = sift_desc_extractor.compute(img, cv_kpts2)
with Timer(name='FREAK Descriptors'):
    (freak_cv_kpts, freak_cv_desc) = freak_desc_extractor.compute(img, cv_kpts2)
# Draw
print('Draw')
figure(1)
freak_image = cv2.drawKeypoints(img, freak_cv_kpts,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imshow(freak_image)

figure(2)
sift_image = cv2.drawKeypoints(img, sift_cv_kpts,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imshow(sift_image)

show()
# ### Reminder:
#       cv2.FlannBasedMatcher()
# Convert

"""
print('Convert')
def convert_cv_kpts_to_storage_represenation(cv_kpts, kpts_dtype=np.float32):
   '''converts opencv keypoints to the [x,y,a,c,d] ellipse format
       outaut has type: ftps_dtype=np.float32'''
    # Convert points
    xy_list = np.array([cv_kp.pt for cv_kp in cv_kpts])
    # Self Indulgence: 
    #                  [[np.cos(cv_kp.angle)*tau/360] for cv_kp in cv_kpts]  # tauday.com
    diag_shape_list  = [np.cos(cv_kp.angle)*np.pi/180 for cv_kp in cv_kpts]
    # Convert ellipses
    scale_list = np.array([3.0/((cv_kp.size)**2) for cv_kp in cv_kpts])
    shape_list = np.array([np.array((s, 0, s)) for s in scale_list])
    #octave_list = [cv_kp.octave for cv_kp in cv_kpts]
    kpts_ = np.hstack((xy_list, shape_list))
    kpts = np.array(kpts_, dtype=kpts_dtype)
    # convert to storage represntation
    kpts_ = np.hstack((xy_list, shape_list))
    kpts = np.array(kpts_, dtype=kpts_dtype)
    return kpts
kpts = convert_cv_kpts_to_storage_represenation(cv_kpts)
"""
