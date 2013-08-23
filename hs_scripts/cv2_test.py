from PIL import Image
from pylab import *
import cv2
import matplotlib.cbook as cbook
import numpy as np

#---------------
# Sample Data
#---------------
# three images from matplotlibs sample data
grace_file = cbook.get_sample_data('grace_hopper.jpg')
lena_file  = cbook.get_sample_data('lena.png')
ada_file   = cbook.get_sample_data('ada.png')
grace = Image.open(grace_file)
lena  = Image.open(lena_file)
ada   = Image.open(ada_file)

# two arrays of correspondings float32 points
pts1 = np.array([(0,0),   (100,100), (0,100)],  dtype=float)
pts2 = np.array([(10,10), (120,120), (10,100)], dtype=float)
#---------------

#---------------
# Test Functions 
def show_data():
    figure(1)
    subplot(131)
    imshow(grace)
    subplot(132)
    imshow(lena)
    subplot(133)
    imshow(ada)

def homogonize(pts):
    'adds a dimension of ones to create homogonized coordinates'
    return np.hstack([pts, np.ones((len(pts),1))])

def test_affine():
    '''Finds an affine transformation between two sets of corresponding points
    The points must be of shape (2xN) and of datatype float32 '''
    # Get affine tranform
    M = cv2.getAffineTransform(np.float32(pts1), np.float32(pts2))
    # Apply affine tranform to homogonized (and transposed) coordinates 
    pts1_transformed = (M.dot(homogonize(pts1).T)).T
    # The affine transformation should move pts1 space to pts2 space
    passed = all(pts1_transformed - pts2 < 1E-9)
    return passed

#----------------


