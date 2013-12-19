#!/usr/bin/env python
from __future__ import division, print_function
from os.path import dirname, realpath, join, exists, normpath
import os
import shutil
import helpers
import sys

INSTALL_REQUIRES = [
    'numpy>=1.5.0',
    'scipy>=0.7.2',
    'PIL>=1.1.7'
]

MODULES = [
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
INSTALL_OPTIONAL = [
    'python-qt>=.50',
    'matplotlib>=1.2.1rc1',
    #'pyvlfeat>=0.1.1a3'
]

INSTALL_OTHER = [
    'boost-python>=1.52',
    'Cython>=.18',
    'ipython>=.13.1'
]

INSTALL_BUILD = [
    'Cython'
]

INSTALL_DEV =  [
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


def build_pyinstaller():
    cwd = normpath(realpath(dirname(__file__)))
    print('Current working directory: %r' % cwd)
    build_dir = join(cwd, 'build')
    dist_dir = join(cwd, 'dist')
    assert exists('setup.py'), 'must be run in hotspotter directory'
    assert exists('../hotspotter/setup.py'), 'must be run in hotspotter directory'
    assert exists('_setup'), 'must be run in hotspotter directory'
    for rmdir in [build_dir, dist_dir]:
        if exists(rmdir):
            helpers.remove_file(rmdir)
    os.system('pyinstaller _setup/pyinstaller-hotspotter.spec')

    if sys.platform == 'darwin' and exists("dist/HotSpotter.app/Contents/"):
        shutil.copyfile("_setup/hsicon.icns", "dist/HotSpotter.app/Contents/Resources/icon-windowed.icns")
        shutil.copyfile("_setup/Info.plist", "dist/HotSpotter.app/Contents/Info.plist")


def compile_ui():
    'Compiles the qt designer *.ui files into python code'
    pyuic4_cmd = {'win32':  'C:\Python27\Lib\site-packages\PyQt4\pyuic4',
                  'linux2': 'pyuic4',
                  'darwin': 'pyuic4'}[sys.platform]
    widget_dir = join(dirname(realpath(__file__)), '_frontend')
    print('Compiling qt designer files in %r' % widget_dir)
    for widget_ui in helpers.glob(widget_dir, '*.ui'):
        widget_py = os.path.splitext(widget_ui)[0] + '.py'
        cmd = ' '.join([pyuic4_cmd, '-x', widget_ui, '-o', widget_py])
        print('compile_ui()>' + cmd)
        os.system(cmd)


def clean():
    assert exists('setup.py'), 'must be run in hotspotter directory'
    assert exists('../hotspotter/setup.py'), 'must be run in hotspotter directory'
    assert exists('_setup'), 'must be run in hotspotter directory'
    cwd = normpath(realpath(dirname(__file__)))
    print('Current working directory: %r' % cwd)
    helpers.delete(join(cwd, 'dist'))
    helpers.delete(join(cwd, 'build'))
    helpers.delete(join(cwd, "'"))  # idk where this file comes from


def fix_tpl_permissions():
    os.system('chmod +x _tpl/extern_feat/*.mac')
    os.system('chmod +x _tpl/extern_feat/*.ln')


if __name__ == '__main__':
    print('Entering HotSpotter setup')
    for cmd in iter(sys.argv[1:]):
        if cmd in ['clean']:
            clean()
            sys.exit(0)
        if cmd in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)
        if cmd in ['build_pyinstaller', 'build_installer']:
            build_pyinstaller()
            sys.exit(0)
