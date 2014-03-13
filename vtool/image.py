# LICENCE
from __future__ import print_function, division
# Science
import cv2


CV2_WARP_KWARGS = {'flags': cv2.INTER_LANCZOS4,
                   'borderMode': cv2.BORDER_CONSTANT}


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def warpAffine(img, M, dsize):
    warped_img = cv2.warpAffine(img, M[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img
