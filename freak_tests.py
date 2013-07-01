import cv2
from pylab import *
from hotspotter.helpers import Timer, figure, myprint
import os
figure(1, doclf=True, title='SIFT Descriptors')
figure(2, doclf=True, title='FREAK Descriptors')

# Define Image
chip_fpath = 'D:/data/work/PZ_flankhack/images/img-0000001.jpg'
# Read Image
img3 = cv2.imread(chip_fpath)
img2  = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img   = cv2.resize(img2,(100,100))
imshow(img)
set_cmap('gray')
params = { 'kpts_type' :'DoG'}
xpmp = {'DoG':'SIFT', 'SIFT':'SIFT'}
# Detect  
print('Detect and Extract')
kpts_detector = cv2.FeatureDetector_create(xpmp[params['kpts_type']])
#kpts_detector = cv2.PyramidAdaptedFeatureDetector(kpts_detector3, maxLevel=1)
#kpts_detector  = cv2.GridAdaptedFeatureDetector(kpts_detector3) # max number of features
with Timer(msg='Detecting '+params['kpts_type']+' keypoints'):
    cv_dog_kpts = kpts_detector.detect(img)  
# Remove Angle
for i in xrange(len(cv_dog_kpts)):
    cv_dog_kpts[i].angle = 0
print('+ Detected  '+str(len(cv_dog_kpts))+' keypoints\n')

# Extract
SIFT_PARAMS = ['contrastThreshold', 'edgeThreshold', 'nFeatures', 'nOctaveLayers', 'sigma']
FREAK_PARAMS = ['nbOctave', 'orientationNormalized', 'patternScale', 'scaleNormalized']
sift_extractor  = cv2.DescriptorExtractor_create('SIFT')
freak_extractor = cv2.DescriptorExtractor_create('FREAK')
freak_extractor.setBool('orientationNormalized', False)

with Timer(msg='Extract '+str(len(cv_dog_kpts))+' SIFT descriptors'):
    (sift_cv_kpts, sift_cv_descs)   = sift_extractor.compute(img, cv_dog_kpts)
print('+SIFT has '+str(len(sift_cv_kpts))+' keypoints\n')

with Timer(msg='Extract '+str(len(cv_dog_kpts))+' FREAK descriptors'):
    (freak_cv_kpts, freak_cv_descs) = freak_extractor.compute(img, cv_dog_kpts)
print('+Freak has '+str(len(freak_cv_kpts))+' keypoints\n')

# Draw
print('Draw')
cv_flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
figure(1)
freak_image = cv2.drawKeypoints(img, freak_cv_kpts, flags=cv_flags)
imshow(freak_image)
figure(2)
sift_image = cv2.drawKeypoints(img, sift_cv_kpts, flags=cv_flags)
imshow(sift_image)

# Draw Individuals
sel = 0

#for siftkp, freakkp in zip(sift_cv_kpts, freak_cv_kpts):
    #myprint(siftkp,'SIFT KP')
    #myprint(freakkp,'FREAK KP')
    #print('-----')

# Select Unoccluded Keypoint
sift_kpt = sift_cv_kpts[sel]
freak_kpt = freak_cv_kpts[sel]
print('Unoccluded keypoints: ')
myprint( sift_kpt, prefix='sift_kpt', indent='  ')
myprint(freak_kpt, prefix='freak_kpt', indent='  ')
# Select Unoccluded Descriptor
sift_desc = sift_cv_descs[sel]
freak_desc = freak_cv_descs[sel]
# Print Unoccluded Keypoint
#kpt2 = cv_dog_kpts[sel]
#myprint(kpt2,prefix='\nOriginal Keypoint', indent='  ')

#mykpt = cv2.KeyPoint(50, 50, _size=20, _angle=0, _response=1, _octave=1)
#myprint(sift_desc, prefix='Extracting one descriptor from: ')
#(sift_kpt2,  sift_desc2) = map(lambda a: a[0], sift_extractor.compute(img, [sift_kpt]))

#(freak_mykpts2, freak_mydesc2) = map(lambda a: a[0], freak_extractor.compute(img, [freak_kpt]))

#with Timer(msg='Computing one SIFT descriptor'):
#with Timer(msg='Computing one FREAK descriptor'):
#freak_extractor2 = cv2.DescriptorExtractor_create('FREAK')
#freak_extractor2.setBool('orientationNormalized', False)
#(freak_kpt2, freak_desc2) = freak_extractor2.compute(img, [freak_kpt, sift_kpt])
    #(freak_kpt2, freak_desc2) = freak_extractor.compute(img, [freak_kpt])
    #(freak_kpt2, freak_desc2) = freak_extractor.compute(img, cv_dog_kpts)


#myprint(sift_kpt2, prefix='sift_kpt2 ')
#myprint(freak_kpt2, prefix='freak_kpt2 ')

#show()
