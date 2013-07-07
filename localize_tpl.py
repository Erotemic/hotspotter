# TODO: Let pyinstaller take care of this

#os.chdir(HOTSPOTTER_INSTALL_PATH)
#import hotspotter.tpl.cv2 as cv2 
import shutil
import os, sys
from hotspotter.helpers import checkpath, vd
from os.path import join
import types
def copy_all_files(src_dir, dest_dir, glob_str_list):
    from fnmatch import fnmatch
    if type(glob_str_list) != types.ListType:
        glob_str_list = [glob_str_list]
    for _fname in os.listdir(src_dir):
        for glob_str in glob_str_list:
            if fnmatch(_fname, glob_str):
                src_path = os.path.normpath(join(src_dir, _fname))
                dest_path = os.path.normpath(join(dest_dir, _fname))
                if os.path.exists(dest_path):
                    sys.stdout.write('!!! Overwriting and ')
                sys.stdout.write('Copying: \n ...'+src_path+' \n ... -> '+dest_path+'\n')
                shutil.copy(src_path, dest_path)
                break

SITEPACKAGES_PATH = 'C:/Python27/Lib/site-packages'
HOME = 'C:/Users/jon.crall'
HOTSPOTTER_INSTALL_PATH = HOME+'/code/hotspotter'
OPENCV_INSTALL_PATH= r'C:\Program Files (x86)\OpenCV'

HS_TPL_ROOT = HOTSPOTTER_INSTALL_PATH+'/hotspotter/tpl'
HS_TPL_LIB  = HS_TPL_ROOT+'/lib/'+sys.platform

lib_exts = ['*.dll', '*.a']
pylib_exts = ['*.pyd']

def localize_opencv():
    OPENCV_BUILD = HOME+'/code/opencv/build'
    OPENCV_BIN = OPENCV_BUILD+'/bin'
    OPENCV_LIB = OPENCV_BUILD+'/lib'
    OPENCV_TPL = OPENCV_BUILD+'/3rdparty/lib'

    copy_all_files(OPENCV_LIB, HS_TPL_LIB, lib_exts)
    copy_all_files(OPENCV_BIN, HS_TPL_LIB, lib_exts)
    copy_all_files(OPENCV_TPL, HS_TPL_LIB, lib_exts)

    copy_all_files(OPENCV_LIB, HS_TPL_ROOT, pylib_exts)
    copy_all_files(OPENCV_BIN, HS_TPL_ROOT, pylib_exts)
    copy_all_files(OPENCV_TPL, HS_TPL_ROOT, pylib_exts)

def localize_flann():
    FLANN_BUILD  = HOME+'/code/flann/build'
    FLANN_LIB    = FLANN_BUILD+'/lib'
    copy_all_files(FLANN_LIB,  HS_TPL_LIB, lib_exts)

localize_all = False
if 'flann' in sys.argv or localize_all:
    localize_flann()
if 'opencv' in sys.argv or localize_all:
    localize_opencv()
