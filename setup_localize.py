# TODO: Let pyinstaller take care of this

#os.chdir(hotspotter_root)
#import hotspotter.tpl.cv2 as cv2 
import shutil
import os, sys
from hotspotter.helpers import checkpath, vd, ensurepath
from os.path import join
import types
from fnmatch import fnmatch


# A script for windows to install pkg config (needed for building with opencv I think)
def fix_mingw_pkgconfig():
    fix_mingw_pkgconfig_bat = r'''
set pkgconfig_name=pkg-config-lite-0.28-1
set pkgconfig_zip=%pkgconfig_name%-win32.zip
set MINGW_BIN="C:\MinGW\bin"
set MINGW_SHARE="C:\MinGW\bin"
set pkg_config_dlsrc=http://downloads.sourceforge.net/project/pkgconfiglite/0.28-1/%pkgconfig_name%_bin-win32.zip

:: Download pkg-config-lite
wget %pkg_config_dlsrc%

:: Unzip and remove zipfile
unzip %pkgconfig_zip%
rm %pkgconfig_zip%

:: Install contents to MSYS
cp %pkgconfig_name%/bin/pkg-config.exe %MINGW_BIN%
cp -r %pkgconfig_name%/share/aclocal %MINGW_SHARE%
'''
    os.system(fix_mingw_pkgconfig_bat)

def copy(src_path, dest_path):
    if os.path.exists(dest_path):
        sys.stdout.write('!!! Overwriting and ')
    sys.stdout.write('Copying: \n ...'+src_path+' \n ... -> '+dest_path+'\n')
    shutil.copy(src_path, dest_path)

def copy_all_files(src_dir, dest_dir, glob_str_list):
    if type(glob_str_list) != types.ListType:
        glob_str_list = [glob_str_list]
    for _fname in os.listdir(src_dir):
        for glob_str in glob_str_list:
            if fnmatch(_fname, glob_str):
                src_path = os.path.normpath(join(src_dir, _fname))
                dest_path = os.path.normpath(join(dest_dir, _fname))
                copy(src_path, dest_path)
                break


__HOME__ = os.path.expanduser("~")
__CODE__ = os.path.expanduser("~")+'/code'

hotspotter_root = __CODE__+'/hotspotter'
tpl_root = hotspotter_root+'/hotspotter/tpl'

lib_exts = ['*.dll', '*.a']
pylib_exts = ['*.pyd']

if sys.platform == 'win32':
    cmake_cmd = 'cmake -G "MSYS Makefiles" '
else:
    cmake_cmd = 'cmake -G "Unix Makefiles" '

install_prefix = '/usr/local/'
py_dist_packages = install_prefix+'/lib/python2.7/dist-packages'
if sys.platform == 'win32':
    install_prefix= r'C:\Program Files (x86)\OpenCV'
    py_dist_packages = r'C:\Python27\Lib\site-packages'


def __cmd(cmd, *args):
    if 'print' in args: 
        print('...Running command: '+cmd)
    ret = os.system(cmd)
    if 'print' in args: 
        print('...In reference to command: '+str(cmd))
        print('...Return code: '+str(ret))
    if 'raise' in args:
        if ret != 0: 
            print('...ERROR!')
            ex_msg = '...In reference to command: '+cmd+\
                '\n...Return code: '+str(ret)
            raise Exception(ex_msg)
        else:
            print('...SUCCESS!')
    return ret

def __sudo_cmd(cmd, *args):
    if sys.platform == 'win32':
        ret = __cmd(cmd, *args)
    else:
        ret = __cmd('sudo '+cmd, *args)
    return ret

# Header print
#def hprint(msg, lvl=0, topline='='):
    #print('\n\n'+'='*len(init_str))

def cmake_flags2str(cmake_flags):
    'Turns a dict into a cmake arguments'
    cmake_arg_str = ''
    for key, val in cmake_flags.iteritems():
        if val == True:
            val = 'ON'
        if val == False:
            val = 'OFF'
        cmake_arg_str+='-D'+key+'='+val+' '
    return cmake_arg_str


def cd(dir):
    print('Changing directory to '+dir)
    os.chdir(dir)

def __build(pkg_name, branchname='hotspotter_branch', cmake_flags={}, noinstall=False):
    from hs_setup.git_helpers import git_branch, git_version, git_fetch_url
    ''' 
    Generic build function for hotspotter third party libraries: 
        All libraries should be hosted under github.com:Erotemic/<pkg_name>

    '''
    cmd_args = ['raise', 'print']
    # ____ INIT ____
    init_str = '_____ Python is building: '+ pkg_name +' _____'
    print('\n\n'+'='*len(init_str))
    print(init_str)
    code_src = __CODE__+'/'+pkg_name
    code_build = code_src+'/build'
    # ____ CHECK SOURCE ____
    print('\n --- Checking code source dir: '+code_src+'\n')
    if not checkpath(code_src): 
        if not checkpath(__CODE__):
            raise Exception('We have problems')
        cd(__CODE__)
        __cmd('git clone git@github.com:Erotemic/'+pkg_name+'.git', *cmd_args)
    cd(code_src)

    # ____ CHECK GIT ____
    print('\n --- Checking git info')
    current_branch = git_branch()
    fetch_url = git_fetch_url()
    version = git_version()
    print('   * fetch_url='+str(fetch_url))
    print('   * branch='+str(current_branch))
    #print('  * version='+str(version))
    if current_branch != branchname:
        __cmd('git checkout '+branchname, *cmd_args)

    # ____ CHECK BUILD ____
    print('\n --- Creating build dir: ' + code_build + '\n')
    ensurepath(code_build)
    cd(code_build)

    # ____ CMAKE ____
    print('\n --- Running cmake\n')
    if 'CMAKE_INSTALL_PREFIX' in cmake_flags: 
        raise Exception('Unexpected behavior may be occuring. Overwriting CMAKE_INSTALL_PREFIX')
    user_cmake_args = cmake_flags2str(cmake_flags)
    _cmake_args = '-DCMAKE_INSTALL_PREFIX='+install_prefix+' '+user_cmake_args
    _cmake_args = _cmake_args.replace('\n',' ')
    __cmd(cmake_cmd + _cmake_args + ' ..', *cmd_args)

    # ____ MAKE ____
    print('\n --- Running make\n') 
    __cmd('make -j9', *cmd_args)

    # ____ INSTALL ____
    if noinstall:
        print('\n --- Not Installing\n')
    else:
        print('\n --- Installing to: '+install_prefix+'\n')
        __sudo_cmd('make install', *cmd_args)

    # ____ END ____
    cd(hotspotter_root)
    exit_msg =  ' --- Finished building: '+pkg_name
    print('\n'+exit_msg)
    print('='*len(exit_msg)+'\n')

def build_hesaff():
    __build('hesaff', branchname='hotspotter_branch', noinstall=True)

def build_opencv():
    __build('opencv', branchname='freak_modifications')

def build_flann():
    cmake_flags = { 'BUILD_MATLAB_BINDINGS' : False }
    if sys.platform == 'win32':
        cmake_flags['CMAKE_BUILD_TYPE']   = 'Release'
        cmake_flags['CMAKE_C_FLAGS']   = '-m32'
        cmake_flags['CMAKE_CXX_FLAGS'] = '-m32'
        cmake_flags['USE_OPENMP'] = False
    __build('flann', branchname='hotspotter_flann')
    
def localize_hesaff():
    print('____ Localizing hessaff ____')
    hesaff_build = __CODE__+'/hesaff/build'
    checkpath(hesaff_build)
    tpl_hesaff = tpl_root + '/hesaff'
    ensurepath(tpl_hesaff) 
    copy_all_files(hesaff_build, tpl_hesaff, 'hesaff*')
    pass

def localize_opencv():
    print('____ Localizing opencv ____')
    # Where to install
    tpl_cv2 = tpl_root + '/cv2'
    ensurepath(tpl_cv2) 
    # Libraries
    opencv_lib = install_prefix+'/lib'
    copy_all_files(opencv_lib, tpl_cv2, 'libopencv*')
    # Python bindings
    copy_all_files(py_dist_packages, tpl_cv2, 'cv2.so')
    with open(tpl_cv2+'/__init__.py', 'w') as cv2_init:
        cv2_init.write('from cv2 import *')

def localize_flann():
    print('____ Localizing flann ____')
    # Where to install
    # Where to install
    tpl_pyflann   = tpl_root+'/pyflann'
    ensurepath(tpl_pyflann) 
    # Libraries
    flann_lib   = install_prefix+'/lib'
    copy_all_files(flann_lib,  tpl_pyflann, 'libflann*')
    # Python bindings
    pyflann_dir = install_prefix+'/share/flann/python/pyflann'
    copy_all_files(pyflann_dir,  tpl_pyflann, '*.py')


exec_str_template = """
if 'localize_%s' in sys.argv or localize_all:
    localize_%s()
if 'build_%s' in sys.argv or build_all:
    build_%s()
if '%s' in sys.argv:
    localize_%s()
    build_%s()
"""
num_subs = exec_str_template.count('%s')

if __name__ == '__main__':
    localize_all = False
    build_all = False

    tpl_list = ['flann', 'opencv', 'hesaff']

    if 'localize_all' in sys.argv: 
        localize_all = True
    if 'build_all' in sys.argv: 
        build_all = True

    for tpl_name in tpl_list:
        exec_str = exec_str_template % tuple([tpl_name]*num_subs)
        exec(exec_str)
