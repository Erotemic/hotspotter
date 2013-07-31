#!/usr/bin/env python

from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
from os.path import dirname, realpath, join, exists, normpath, isdir, isfile, normpath
from hs_setup.git_helpers import *
from hs_setup.configure import *
import os
import shutil
import subprocess
import sys

INSTALL_REQUIRES = \
[
    'numpy>=1.5.0',
    'scipy>=0.7.2',
    'PIL>=1.1.7'
]

MODULES = \
[
    'sip',
    'PyQt4',
    'PyQt4.Qt',
    'PIL.Image',
    'PIL.PngImagePlugin',
    'PIL.JpegImagePlugin',
    'PIL.GifImagePlugin',
    'PIL.PpmImagePlugin',
    'matplotlib',
    'numpy',
    'scipy',
    'PIL'
]

INSTALL_OPTIONAL = \
[
    'python-qt>=.50',
    'matplotlib>=1.2.1rc1',
    #'pyvlfeat>=0.1.1a3'
]

INSTALL_OTHER = \
[
    'boost-python>=1.52',
    'Cython>=.18',
    'ipython>=.13.1'
]

INSTALL_BUILD = \
[
    'Cython'
]

INSTALL_DEV = \
[
    'py2exe>=0.6.10dev',
    'pyflakes>=0.6.1',
    'pylint>=0.27.0',
    'RunSnakeRun>=2.0.2b1',
    'maliae>=0.4.0.final.0',
    'pycallgraph>=0.5.1'
    'coverage>=3.6'
]

CLASSIFIERS = '''\
Development Status :: 1 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: GPL License
Programming Language :: Python
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
'''
NAME                = 'HotSpotter'
AUTHOR              = 'Jonathan Crall, RPI'
AUTHOR_EMAIL        = 'hotspotter.ir@gmail.com'
MAINTAINER          = AUTHOR
MAINTAINER_EMAIL    = AUTHOR_EMAIL
DESCRIPTION         = 'Image Search for Large Animal Databases.'
LONG_DESCRIPTION    = open('DESCRIPTION.txt').read()
URL                 = 'http://www.cs.rpi.edu/~cralljp'
DOWNLOAD_URL        = 'https://github.com/Erotemic/hotspotter/archive/release.zip'
LICENSE             = 'GPL'
PLATFORMS           = ['Windows', 'Linux', 'Mac OS-X']
MAJOR               = 0
MINOR               = 0
MICRO               = 0
SUFFIX              = ''  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False
VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = join(where,name)
            if ('.' not in name and isdir(fn) and
                isfile(join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out

def write_text(filename, text, mode='w'):
    with open(filename, mode='w') as a:
        try:
            a.write(text)
        except Exception as e:
            print(e)

def write_version_py(filename=join('hotspotter', 'generated_version.py')):
    cnt = ''' # THIS FILE IS GENERATED FROM HOTSPOTTER SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
git_revision = '%(git_revision)s'
full_version = '%(version)s.dev-%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
'''
    FULL_VERSION = VERSION
    if isdir('.git'):
        GIT_REVISION = git_version()
    elif exists(filename):
        # must be a source distribution, use existing version file
        GIT_REVISION = 'RELEASE'
    else:
        GIT_REVISION = 'unknown-git'

    FULL_VERSION += '.dev-' + GIT_REVISION
    text = cnt % {'version': VERSION,
                  'full_version': FULL_VERSION,
                  'git_revision': GIT_REVISION,
                  'isrelease': str(ISRELEASED)}
    write_text(filename, text)


def ensure_findable_windows_dlls():
    numpy_core = r'C:\Python27\Lib\site-packages\numpy\core'
    numpy_libs = ['libiomp5md.dll', 'libifcoremd.dll', 'libiompstubs5md.dll', 'libmmd.dll']
    pydll_dir  = r'C:\Python27\DLLs'

    for nplib in numpy_libs:
        dest = join(pydll_dir, nplib)
        if not exists(dest):
            src = join(numpy_core, nplib)
            shutil.copyfile(src, dest)

    zmqpyd_target = r'C:\Python27\DLLs\libzmq.pyd'
    if not exists(zmqpyd_target):
        #HACK http://stackoverflow.com/questions/14870825/
        #py2exe-error-libzmq-pyd-no-such-file-or-directory
        pyzmg_source = r'C:\Python27\Lib\site-packages\zmq\libzmq.pyd'
        shutil.copyfile(pyzmg_source, zmqpyd_target)

def get_hotspotter_datafiles():
    'Build the data files used by py2exe and py2app'
    import matplotlib
    data_files = []
    # Include Matplotlib data (for figure images and things)
    data_files.extend(matplotlib.get_py2exe_datafiles())
    # Include TPL Libs
    plat_tpllibdir = join('hotspotter', 'tpl','lib', sys.platform)
    if sys.platform == 'win32':
        # Hack to get MinGW dlls in for FLANN
        data_files.append(('',[join(plat_tpllibdir, 'libgcc_s_dw2-1.dll'),
                               join(plat_tpllibdir,'libstdc++-6.dll')]))
    if sys.platform == 'darwin':
        pass
    else:
        for root,dlist,flist in os.walk(plat_tpllibdir):
            tpl_dest = root
            tpl_srcs = [realpath(join(root,fname)) for fname in flist]
            data_files.append((tpl_dest, tpl_srcs))
    # Include Splash Screen
    splash_dest = normpath('hotspotter/front')
    splash_srcs = [realpath('hotspotter/front/splash.png')]
    data_files.append((splash_dest, splash_srcs))
    return data_files

def package_application():
    write_version_py()
    # ---------
    setup_kwargs = get_info_setup_kwarg()
    # ---------
    # Force inclusion the modules that may have not been explicitly included
    includes_modules = MODULES
    # Import whatever database module you can
    packed_db_module = False
    DB_MODULES = ['dbhash', 'gdbm', 'dbm', 'dumbdbm']
    for dbmodule in DB_MODULES:
        try:
            __import__(dbmodule)
            includes_modules.append(dbmodule)
        except ImportError:
            pass
    # --------
    # Get Data Files 
    data_files = get_hotspotter_datafiles()
    setup_kwargs.update({'data_files' : data_files})
    # ---------
    run_with_console = True
    py2_appexe_universal = {
            'optimize'     : 0, # 0,1,2
            'includes'     : includes_modules
    }
    # WINDOWS EXECUTABLE SETUP
    if sys.platform == 'win32':
        ensure_findable_windows_dlls()
        # ---------
        
        # Construct py2exe options
        py2exe_options = py2_appexe_universal
        py2exe_options.update({
            'unbuffered'   : True,
            'skip_archive' : True, #do not place Python bytecode files in an
                                #archive, put them directly in the file system
            'compressed'   : False, #(boolean) create a compressed zipfile
            'bundle_files' : 3 #1=all, 2=all-Interpret, 3=dont bundle
        })
        setup_options={'py2exe' : py2exe_options}
        run_cmd = [{'script': 'main.py',
                    'icon_resources': [(0, 'hsicon.ico')]}]
        run_type = 'console' if run_with_console else 'windows'
    # 
    # MAC APPLICATION SETUP
    if sys.platform == 'darwin':
        import py2app
        # Construct py2app options
        setup_kwargs.update({'setup_requires':['py2app']})
        py2app_options = py2_appexe_universal
        py2app_options.update({
            'argv_emulation': False,
            'iconfile':'hsicon.icns',
            'plist': {'CFBundleShortVersionString':'1.0.0',}
        })
        py2app_options.update(py2_appexe_universal)
        setup_options={'py2app' : py2app_options}
        run_type = 'app'
        run_cmd = ['main.py']
    # add windows/mac stuff to setup keyword arguments
    setup_kwargs.update({run_type : run_cmd})
    setup_kwargs.update({'options' : setup_options})
    # ---------
    # 
    # Do actual setup
    print 'Running package setup with args: '
    for key, val in setup_kwargs.iteritems():
        print(key+' : '+repr(val))
    setup(**setup_kwargs)

    if sys.platform == 'darwin':
        subprocess.call(["cp", "-r", "hotspotter/tpl/lib/darwin", "dist/HotSpotter.app/Contents/Resources/lib/"])

def get_info_setup_kwarg():
    return dict(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=' '.join([
            'hotsoptter', 'vision', 'animals', 'object recognition',
            'instance recognition', 'naive bayes' ]))

def get_system_setup_kwargs():
    return dict(
        platforms=PLATFORMS,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        install_optional=INSTALL_OPTIONAL,
    )


def build_pyinstaller():
    import os
    cwd = normpath(realpath(dirname(__file__)))
    print cwd
    build_dir = join(cwd, 'build')
    dist_dir = join(cwd, 'dist')
    for rmdir in [build_dir, dist_dir]:
        if exists(rmdir):
            print('Removing '+rmdir)
            os.system('rm -rf '+rmdir)
    os.system('pyinstaller hs_setup/pyinstaller-hotspotter.spec')

    if sys.platform == 'darwin':
        shutil.copyfile("hs_setup/hsicon.icns", "dist/HotSpotter.app/Contents/Resources/icon-windowed.icns")
        shutil.copyfile("hs_setup/Info.plist", "dist/HotSpotter.app/Contents/Info.plist")

if __name__ == '__main__':
    import sys
    print 'Entering HotSpotter setup'
    for cmd in iter(sys.argv[1:]):
        if cmd == 'setup_boost':
            setup_boost()
            sys.exit(0)
        if cmd in ['fix_issues', 'configure']:
            configure()
            sys.exit(0)
        if cmd == 'compile_widgets':
            compile_widgets()
            sys.exit(0)
        if cmd in ['build_pyinstaller', 'build_installer']:
            build_pyinstaller()
            sys.exit(0)
        if cmd in ['localize', 'setup_localize.py']:
            from setup_localize import *
    package_application()
