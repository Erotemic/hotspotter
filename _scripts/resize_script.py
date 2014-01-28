#!/usr/bin/env python
from __future__ import print_function, division
from PIL import Image
from os.path import splitext, join, relpath, split, exists
import cv2
import fnmatch
import numpy as np
import os
import sys

cv2_flags = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)[4]
print(cv2_flags)
cv2_borderMode  = cv2.BORDER_CONSTANT
cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}

__IMG_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm']
__LOWER_EXTS = [ext.lower() for ext in __IMG_EXTS]
__UPPER_EXTS = [ext.upper() for ext in __IMG_EXTS]
IMG_EXTENSIONS =  set(__LOWER_EXTS + __UPPER_EXTS)


def imread(img_fpath):
    try:
        img = Image.open(img_fpath)
        img = np.asarray(img)
    except Exception as ex:
        print('[io] Caught Exception: %r' % ex)
        print('[io] ERROR reading: %r' % (img_fpath,))
        raise
    return img


def build_transform(x, y, w, h, w_, h_, theta, homogenous=False):
    sx = (w_ / w)  # ** 2
    sy = (h_ / h)  # ** 2
    cos_ = np.cos(-theta)
    sin_ = np.sin(-theta)
    tx = -(x + (w / 2))
    ty = -(y + (h / 2))

    T1 = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1]], np.float64)

    S = np.array([[sx, 0,  0],
                  [0, sy,  0],
                  [0,  0,  1]], np.float64)

    R = np.array([[cos_, -sin_, 0],
                  [sin_,  cos_, 0],
                  [   0,     0, 1]], np.float64)

    T2 = np.array([[1, 0, (w_ / 2)],
                   [0, 1, (h_ / 2)],
                   [0, 0, 1]], np.float64)

    M = T2.dot(R.dot(S.dot(T1)))
    #.dot(R)#.dot(S).dot(T2)
    if homogenous:
        transform = M
    else:
        transform = M[0:2, :] / M[2, 2]
    return transform


def resample_image(img_path, new_size, chip_path=None, output_dir=None):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    np_img = imread(img_path)
    # Build transformation
    (gh, gw) = np_img.shape[0:2]
    roi = (0, 0, gw, gh)
    theta = 0
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    Aff = build_transform(rx, ry, rw, rh, rw_, rh_, theta)
    # Rotate and scale
    #pil_img  = Image.open(img_path)
    #pil_chip = pil_img.resize(new_size, Image.ANTIALIAS)
    chip = cv2.warpAffine(np_img, Aff, (rw_, rh_), **cv2_warp_kwargs)
    pil_chip = Image.fromarray(chip)
    # Build a new name
    if chip_path is None:
        name, ext = splitext(img_path)
        ext = ext.lower()
        if not ext in IMG_EXTENSIONS:
            ext = '.png'
        if ext == '.jpg':
            ext = '.jpeg'
        #suffix = '_sz' + repr(new_size).replace(' ', '')
        suffix = '_' + str(cv2_flags)
        chip_path = name + suffix + ext
        #chip_path = name + ext
    # put the chip in the chosen directory
    if output_dir is not None:
        chip_dir, chip_name = split(chip_path)
        chip_path = join(output_dir, chip_name)
    img_format = ext[1:].upper()
    print(chip_path)
    pil_chip.save(chip_path, img_format)


def matches_image(fname):
    fname_ = fname.lower()
    img_pats = ['*' + ext for ext in IMG_EXTENSIONS]
    return any([fnmatch.fnmatch(fname_, pat) for pat in img_pats])


def list_images(img_dpath, ignore_list=[], recursive=False, fullpath=True):
    ignore_set = set(ignore_list)
    gname_list_ = []
    # Get all the files in a directory recursively
    for root, dlist, flist in os.walk(img_dpath):
        for fname in iter(flist):
            gname = join(relpath(root, img_dpath), fname).replace('\\', '/').replace('./', '')
            if fullpath:
                gname_list_.append(join(root, gname))
            else:
                gname_list_.append(gname)
        if not recursive:
            break
    # Filter out non images or ignorables
    gname_list = [gname for gname in iter(gname_list_)
                  if not gname in ignore_set and matches_image(gname)]
    return gname_list


def ensurepath(path_):
    if not exists(path_):
        print('[helpers] mkdir(%r)' % path_)
        os.makedirs(path_)
    return True


if __name__ == '__main__':
    usage = './resize_script input_dir [output_dir]'

    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
        if len(sys.argv) >= 3:
            output_dir = sys.argv[2]
            ensurepath(output_dir)
        else:
            output_dir = None
    else:
        print('Usage: %s' % usage)
        sys.exit(0)

    if not exists(input_dir):
        print('Usage: %s' % usage)
        sys.exit(0)

    new_size = (256, 256)
    print('input_dir = %r' % input_dir)
    print('output_dir = %r' % output_dir)
    print('new_size = %r' % (new_size,))
    print('')
    image_list = list_images(input_dir)
    for image_fpath in image_list:
        print('---------------------')
        print('Resizing %r' % image_fpath)
        resample_image(image_fpath, new_size, output_dir=output_dir)
