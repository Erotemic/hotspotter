# TODO: Let pyinstaller take care of this

#os.chdir(hotspotter_root)
import os, sys
from hotspotter.util import check_path, ensure_path, vd, copy_all, copy
from os.path import join
import types
import textwrap


__HOME__ = os.path.expanduser("~")
__CODE__ = os.path.expanduser("~")+'/code'

hotspotter_root = __CODE__+'/hotspotter'
tpl_root = hotspotter_root+'/hotspotter/tpl'

# Platform compatibility
if sys.platform == 'win32':
    cmake_cmd = 'cmake -G "MSYS Makefiles" '
    make_cmd  = 'make -j'
    # Lets make -j work with msys on windows
    os.system.environ['SHELL']='cmd.exe'
    install_prefix   = os.environ['PROGRAMFILES']
    pypackages = 'C:/Python27/Lib/site-packages'
else:
    cmake_cmd = 'cmake -G "Unix Makefiles" '
    make_cmd  = 'make -j9'
    install_prefix   = '/usr/local'
    pypackages = install_prefix+'/lib/python2.7/dist-packages'

# robust os.system wrapper
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

# os.system + admin privileges
def __sudo_cmd(cmd, *args):
    if sys.platform == 'win32':
        ret = __cmd(cmd, *args)
    else:
        ret = __cmd('sudo '+cmd, *args)
    return ret

# Helper function. Turns dict into valid cmake.exe flags
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

# Wrapper for os.chdir
def cd(dir):
    print('Changing directory to '+dir)
    os.chdir(dir)

# Generic build function
def __build(pkg_name,
            branchname='hotspotter_branch',
            cmake_flags={},
            noinstall=False,
            rm_build=False,
            nomake=False):
    ''' 
    Generic build function for hotspotter third party libraries: 
        All libraries should be hosted under github.com:Erotemic/<pkg_name>
    '''
    from hs_setup.git_helpers import git_branch, git_version, git_fetch_url
    cmd_args = ['raise', 'print']
    # ---- INIT ---- #
    init_str = '_____ Python is building: '+ pkg_name +' _____'
    print('\n\n'+'='*len(init_str))
    print(init_str)
    code_src = __CODE__+'/'+pkg_name
    code_build = code_src+'/build'
    # ---- CHECK SOURCE ---- #
    print('\n --- Checking code source dir: '+code_src+'\n')
    if not check_path(code_src): 
        if not check_path(__CODE__):
            raise Exception('We have problems')
        cd(__CODE__)
        __cmd('git clone git@github.com:Erotemic/'+pkg_name+'.git', *cmd_args)
    cd(code_src)
    # ---- CHECK GIT ---- #
    print('\n --- Checking git info')
    current_branch = git_branch()
    fetch_url = git_fetch_url()
    version = git_version()
    print('   * fetch_url='+str(fetch_url))
    print('   * branch='+str(current_branch))
    #print('  * version='+str(version))
    if current_branch != branchname:
        __cmd('git checkout '+branchname, *cmd_args)
    # ---- CHECK BUILD ---- #
    if rm_build:
        print('\n --- Forcing rm build dir: ' + code_build + '\n')
        if check_path(code_build):
            __cmd('rm -rf '+code_build)
    print('\n --- Creating build dir: ' + code_build + '\n')
    ensure_path(code_build)
    cd(code_build)
    # ---- CMAKE ---- #
    print('\n --- Running cmake\n')
    if not 'CMAKE_INSTALL_PREFIX' in cmake_flags: 
        _cm_install_prefix = install_prefix
        if sys.platform == 'win32': _cm_install_prefix += '/' + pkg_name
        cmake_flags['CMAKE_INSTALL_PREFIX'] = _cm_install_prefix
    _cmake_args = cmake_flags2str(cmake_flags).replace('\n',' ')
    __cmd(cmake_cmd + _cmake_args + ' ..', *cmd_args)
    # ---- MAKE ---- #
    print('\n --- Running make\n') 
    __cmd(make_cmd, *cmd_args)
    # ---- INSTALL ---- #
    if noinstall:
        print('\n --- Not Installing\n')
    else:
        print('\n --- Installing to: '+cmake_flags['CMAKE_INSTALL_PREFIX']+'\n')
        __sudo_cmd('make install', *cmd_args)
    # ---- END ---- #
    cd(hotspotter_root)
    exit_msg =  ' --- Finished building: '+pkg_name
    print('\n'+exit_msg)
    print('='*len(exit_msg)+'\n')

def build_hesaff(nomake=False):
    cmake_flags = {}
    if sys.platform == 'win32':
        cmake_flags['CMAKE_C_FLAGS']   = '-march=i486'
        cmake_flags['CMAKE_CXX_FLAGS'] = '-march=i486'
    __build('hesaff',
            branchname='hotspotter_branch',
            cmake_flags=cmake_flags,
            noinstall=True,
            nomake=nomake)

def build_opencv(nomake=False):
    cmake_flags = {
        'BUILD_opencv_gpu'           : False,
        'BUILD_opencv_gpuarithm'     : False,
        'BUILD_opencv_gpubgsegm'     : False,
        'BUILD_opencv_gpucodec'      : False,
        'BUILD_opencv_gpufeatures2d' : False,
        'BUILD_opencv_gpufilters'    : False,
        'BUILD_opencv_gpuimgproc'    : False,
        'BUILD_opencv_gpuoptflow'    : False,
        'BUILD_opencv_gpustereo'     : False,
        'BUILD_opencv_gpuwarping'    : False }
    if sys.platform == 'win32':
        #-march=i486
        # http://software.intel.com/en-us/blogs/2012/09/26/gcc-x86-performance-hints
        #intel_x86_gcc_flags = '-m32 -mfpmath=sse -Ofast -flto -march=native -funroll-loops'
        #intel_x86_gcc_flags = '-m32 -O2 -march=i786'
        #intel_x86_gcc_flags = '-m32 -O2'
        intel_x86_gcc_flags = '-m32 -O2 -mstackrealign'

        # something about the stack with opencv and sse2 on mingw
        #http://code.opencv.org/issues/1932#note-1

        # ok still crashes sometimes
        working_msys_cflags = '-m32 -O1 -DNDEBUGS' #Use SSE2 with this

        #might work
        # -O3 and -mpreferred-stack-boundary=2

        # Think *can* I fixed it using: __attribute__((force_align_arg_pointer))
        # http://www.peterstock.co.uk/games/mingw_sse/
        # added stack align to freak function call

        # WINDOWS DEFAULTS:
        # CMAKE_CXX_FLAGS_DEBUG : -g
        # CMAKE_CXX_FLAGS_MINSIZEREL : -Os -DNDEBUG -g
        # CMAKE_CXX_FLAGS_RELEASE : -O3 -DNDEBUG
        # CMAKE_CXX_FLAGS_RELWITHDEB_INFO : -O2 -DNDEBUG -g
        # CMAKE_C_FLAGS_DEBUG : -g
        # CMAKE_C_FLAGS_MINSIZEREL : -Os -DNDEBUG -g
        # CMAKE_C_FLAGS_RELEASE : -O3 -DNDEBUG
        # CMAKE_C_FLAGS_RELWITHDEB_INFO : -O2 -DNDEBUG -g

        win32_flags = {
            'CMAKE_INSTALL_PREFIX'       : '"'+install_prefix+'/OpenCV"',
            'CMAKE_BUILD_TYPE'           : 'Release',
            'CMAKE_C_FLAGS'              : '-m32',
            'CMAKE_CXX_FLAGS'            : '-m32',
            'ENABLE_SSE'                 : False,
            'ENABLE_SSE2'                : False,
            'ENABLE_SSE3'                : False,
            'ENABLE_SSE41'               : False,
            'ENABLE_SSE42'               : False,
            'ENABLE_SSEE3'               : False,
            'CMAKE_CXX_FLAGS_RELWITHDEBINFO' : '-O2 -g -DNDEBUGS',
            'CMAKE_C_FLAGS_RELWITHDEBINFO'   : '-O2 -g -DNDEBUGS',
            'CMAKE_CXX_FLAGS_RELEASE'        : '-O2 -DNDEBUGS',
            'CMAKE_C_FLAGS_RELEASE'          : '-O2 -DNDEBUGS'}
        # Try setting release type to Debug to get SSE2 to work on win32
        # Also try setting from -O3 to -O2
        #
        # SSE2 Info:
        # http://gruntthepeon.free.fr/ssemath/
        # -mfpmath=sse
        # Info on CFLAGS
        # http://software.intel.com/en-us/blogs/2012/09/26/gcc-x86-performance-hints
        
        # Info on why make crashes on Windows:
        # http://stackoverflow.com/questions/1533425/make-parallel-jobs-on-windows
        # GCC Stable: http://tdm-gcc.tdragon.net/
        cmake_flags.update(win32_flags)
    __build('opencv',
            cmake_flags=cmake_flags,
            branchname='freak_modifications',
            nomake=nomake)

def build_flann(nomake=False):
    cmake_flags = { 
        'BUILD_MATLAB_BINDINGS' : False }
    if sys.platform == 'win32':
        win32_flags = {
            'CMAKE_INSTALL_PREFIX' : '"'+install_prefix+'/Flann"',
            'CMAKE_BUILD_TYPE'     : 'Release',
            'CMAKE_C_FLAGS'        : '-m32',
            'CMAKE_CXX_FLAGS'      : '-m32',
            'USE_OPENMP'           : False,
            'HDF5_INCLUDE_DIRS'    : '',
            'HDF5_ROOT_DIR'        : '' }
        cmake_flags.update_win32_flags
    __build('flann', 
            cmake_flags=cmake_flags,
            branchname='hotspotter_flann',
            nomake=nomake)
    
def localize_hesaff():
    print('____ Localizing hessaff ____')
    hesaff_build = __CODE__ + '/hesaff/build'
    hesaff_pybnd = __CODE__ + '/hesaff/python_bindings'
    check_path(hesaff_build)
    tpl_hesaff = tpl_root + '/hesaff'
    ensure_path(tpl_hesaff) 
    copy_all(hesaff_build, tpl_hesaff, 'hesaff*')
    copy_all(hesaff_pybnd, tpl_hesaff, '*.py')
    os.system('chmod +x ' + tpl_hesaff + '/hesaff*')

def localize_opencv():
    raise Exception('dont do this no more')
    print('____ Localizing opencv ____')
    # Where to install
    tpl_cv2 = tpl_root + '/cv2'
    ensure_path(tpl_cv2) 
    # Libraries
    opencv_lib = install_prefix+'/lib'
    if sys.platform == 'win32':
        # The opencv libraries are in bin not lib on windows. x.x
        opencv_lib = install_prefix+'/OpenCV/bin'
        # Move the MinGW libs too
        mingw_lib = 'C:/MinGW/bin'
        copy_all(mingw_lib, tpl_cv2, ['libgcc_s_dw2-1.dll',
                                    'libstdc++-6.dll'])
    copy_all(opencv_lib, tpl_cv2, 'libopencv*')
    # Python bindings
    copy_all(pypackages, tpl_cv2, ['cv2.so','cv2.pyd','libcv2*'])
    with open(tpl_cv2+'/__init__.py', 'w') as cv2_init:
        cv2_init.write(textwrap.dedent('''
        # autogenerated in build_tpl.py
        import os, sys
        from os.path import realpath, dirname
        tpl_cv2 = realpath(dirname(__file__))
        sys.path.insert(0, tpl_cv2)
        os.environ['PATH'] = tpl_cv2 + os.pathsep + os.environ['PATH']
        try:
            from cv2 import *
        except Exception as ex:                       
            print(repr(ex))
            print(os.environ['PATH'])
            print(sys.path)
            raise
        '''))

def localize_flann():
    print('____ Localizing flann ____')
    # Where to install
    # Where to install
    tpl_pyflann   = tpl_root+'/pyflann'
    ensure_path(tpl_pyflann) 
    if sys.platform == 'win32':
        # Libraries
        flann_lib   = 'C:/Program Files (x86)/flann/lib'
        copy_all(flann_lib,  tpl_pyflann, 'libflann*')
        # Better do the bin as well (like opencv)
        # yups
        flann_bin   = 'C:/Program Files (x86)/flann/bin'
        copy_all(flann_bin,  tpl_pyflann, '*.dll')
        # Python bindings
        pyflann_dir = pypackages+'/pyflann'
        copy_all(pyflann_dir,  tpl_pyflann, '*.py')
    else:
        # Libraries
        flann_lib   = install_prefix+'/lib'
        copy_all(flann_lib,  tpl_pyflann, 'libflann*')
        # Python bindings
        pyflann_dir = install_prefix+'/share/flann/python/pyflann'
        copy_all(pyflann_dir,  tpl_pyflann, '*.py')


exec_str_template = """
if 'localize_%s' in sys.argv or localize_all:
    localize_%s()
if 'build_%s' in sys.argv or build_all:
    build_%s()
if '%s' in sys.argv:
    build_%s()
    localize_%s()
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
