'''
There is an issue with cv2.warpAffine on macs.
This is a test to further investigate the issue.
'''
from __future__ import division, print_function
import os
import sys
from os.path import dirname, join, expanduser, exists, split


def ensure_hotspotter():
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    hotspotter_location = split(hotspotter_dir)[0]
    sys.path.append(hotspotter_location)
ensure_hotspotter()
from hotspotter.dbgimport import *  # NOQA
from hotspotter import chip_compute2 as cc2
from hotspotter import Parallelize as parallel
from hotspotter import helpers
from hotspotter import fileio as io
import cv2
parallel_compute = parallel.parallel_compute

try:
    test_dir = join(dirname(__file__))
except NameError as ex:
    test_dir = os.getcwd()

# Get information
gfpath = helpers.get_lena_fpath()
cfpath = join(test_dir, 'tmp_chip.png')
roi = [0, 0, 100, 100]
new_size = (500, 500)
theta = 0
img_path = gfpath

# parallel tasks
nTasks = 100
gfpath_list = [gfpath] * nTasks
cfpath_list = [cfpath] * nTasks
roi_list = [roi] * nTasks
theta_list = [theta] * nTasks
chipsz_list = [new_size] * nTasks

printDBG = print


def extract_chip(img_path, chip_path, roi, theta, new_size):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    np_img = io.imread(img_path)
    # Build transformation
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    Aff = cc2.build_transform(rx, ry, rw, rh, rw_, rh_, theta, affine=True)
    print('built transform Aff=\n%r' % Aff)
    # Rotate and scale
    #chip = cv2.warpAffine(np_img, Aff, (rw_, rh_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    chip = cv2.warpAffine(np_img, Aff, (rw_, rh_), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    print('warped')
    return chip

# Try one task
cc2.extract_chip(gfpath, cfpath, roi, theta, new_size)

arg_list = [gfpath_list, cfpath_list, roi_list, theta_list, chipsz_list]
pcc_kwargs = {
    'arg_list': arg_list,
    'lazy': True,
    'common_args': []
}
parallel_compute(extract_chip, **pcc_kwargs)
