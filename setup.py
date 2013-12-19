#!/usr/bin/env python
from __future__ import division, print_function
from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
from os.path import dirname, realpath, join, exists, normpath, isdir, isfile, normpath
from _setup import git_helpers as git_helpers
#from hs_scripts.setup.configure as hs_configure
import os
import textwrap
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


#'parse'
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
LONG_DESCRIPTION    = open('_doc/DESCRIPTION.txt').read()
URL                 = 'http://www.cs.rpi.edu/~cralljp'
DOWNLOAD_URL        = 'https://github.com/Erotemic/hotspotter/archive/release.zip'
LICENSE             = 'GNU'
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

def write_version_py(filename=None):
    if filename is None:
        hsdir = os.path.split(realpath(__file__))[0]
        filename = join(hsdir, 'generated_version.py')
    cnt = textwrap.dedent('''
    # THIS FILE IS GENERATED FROM HOTSPOTTER SETUP.PY
    short_version = '%(version)s'
    version = '%(version)s'
    git_revision = '%(git_revision)s'
    full_version = '%(version)s.dev-%(git_revision)s'
    release = %(isrelease)s
    if not release:
        version = full_version''')
    FULL_VERSION = VERSION
    if isdir('.git'):
        GIT_REVISION = git_helpers.git_version()
    # must be a source distribution, use existing version file
    elif exists(filename):
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
    plat_tpllibdir = join('hotspotter','_tpl','lib', sys.platform)
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
    splash_dest = normpath('_frontend')
    splash_srcs = [realpath('_frontend/splash.png')]
    data_files.append((splash_dest, splash_srcs))
    return data_files

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
    print(cwd)
    build_dir = join(cwd, 'build')
    dist_dir = join(cwd, 'dist')
    for rmdir in [build_dir, dist_dir]:
        if exists(rmdir):
            print('Removing '+rmdir)
            os.system('rm -rf '+rmdir)
    os.system('pyinstaller _setup/pyinstaller-hotspotter.spec') 

    if sys.platform == 'darwin':
        shutil.copyfile("_setup/hsicon.icns", "dist/HotSpotter.app/Contents/Resources/icon-windowed.icns")
        shutil.copyfile("_setup/Info.plist", "dist/HotSpotter.app/Contents/Info.plist")

import helpers
def compile_ui():
    'Compiles the qt designer *.ui files into python code'
    pyuic4_cmd = {'win32'  : 'C:\Python27\Lib\site-packages\PyQt4\pyuic4',
                  'linux2' : 'pyuic4',
                  'darwin' : 'pyuic4'}[sys.platform]
    widget_dir = join(dirname(realpath(__file__)), '_frontend')
    print('Compiling qt designer files in %r' % widget_dir)
    for widget_ui in helpers.glob(widget_dir, '*.ui'):
        widget_py = os.path.splitext(widget_ui)[0]+'.py'
        cmd = pyuic4_cmd+' -x '+widget_ui+' -o '+widget_py
        print('compile_ui()>'+cmd)
        os.system(cmd)

if __name__ == '__main__':
    import sys
    print('Entering HotSpotter setup')
    for cmd in iter(sys.argv[1:]):
        if cmd == 'setup_boost':
            setup_boost()
            sys.exit(0)
        if cmd in ['fix_issues', 'configure']:
            configure()
            sys.exit(0)
        if cmd in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)
        if cmd in ['build_pyinstaller', 'build_installer']:
            build_pyinstaller()
            sys.exit(0)
        if cmd in ['localize', 'setup_localize.py']:
            from setup_localize import *
    #package_application() - moved to graveyard (pyapp_pyexe.py)
