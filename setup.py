#!/usr/bin/env python
from __future__ import division, print_function
from os.path import dirname, realpath, join, exists, normpath, expanduser, splitext
import os
import sys
# Hotspotter
from hscom import helpers

HOME = os.path.expanduser('~')
# Allows other python modules (like hesaff) to find hotspotter modules
sys.path.append(dirname(__file__))

INSTALL_REQUIRES = [
    'numpy>=1.5.0',
    'scipy>=0.7.2',
    'PIL>=1.1.7'
    'argparse>=1.2.1'
    'ipython>=1.1.0'
    'pylru>=1.0.6'
    'pandas>=0.12.0',
    'python-qt>=.50',
    'matplotlib>=1.3.1',
    'scikit-image>=0.9dev'
    'scikit-learn>=0.14a1'
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
LICENSE             = 'Apache'
PLATFORMS           = ['Windows', 'Linux', 'Mac OS-X']
MAJOR               = 0
MINOR               = 0
MICRO               = 0
SUFFIX              = ''  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False
VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def _cd(dpath, verbose=True):
    if verbose:
        print('[setup] change dir to: %r' % dpath)
    os.chdir(dpath)


def _cmd(args, verbose=True, sudo=False):
    sys.stdout.flush()
    import subprocess
    #import shlex
    #if isinstance(args, str):
        #if os.name == 'posix':
            #args = [args]
        #else:
            #args = shlex.split(args)
    if sudo is True and sys.platform != 'win32':
        args = 'sudo ' + args
    PIPE = subprocess.PIPE
    # DANGEROUS: shell=True. Grats hackers.
    print('[setup] Running: %r' % args)
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE, shell=True)
    if verbose:
        ''' KNOWN PYTHON 2.x BUG
        #http://stackoverflow.com/questions/803265/
        #getting-realtime-output-using-subprocess
        for line in proc.stdout.readlines(): '''
        logged_list = []
        append = logged_list.append
        write = sys.stdout.write
        flush = sys.stdout.flush
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            write(line)
            flush()
            append(line)
        out = '\n'.join(logged_list)
        (out_, err) = proc.communicate()
        print(err)
    else:
        # Surpress output
        (out, err) = proc.communicate()
    # Make sure process if finished
    ret = proc.wait()
    return out, err, ret

if sys.platform == 'win32':
    buildscript_fmt = 'mingw_%s_build.bat'
else:
    buildscript_fmt = 'unix_%s_build.sh'


def build_pyinstaller():
    cwd = normpath(realpath(dirname(__file__)))
    print('[setup] Current working directory: %r' % cwd)
    build_dir = join(cwd, 'build')
    dist_dir = join(cwd, 'dist')
    helpers.delete(dist_dir)
    assert exists('setup.py'), 'must be run in hotspotter source directory'
    assert exists('../hotspotter/setup.py'), 'must be run in hotspotter source directory'
    assert exists('../hotspotter/hotspotter'), 'must be run in hotspotter source directory'
    assert exists('_setup'), 'must be run in hotspotter source directory'
    # Remove old files
    for rmdir in [build_dir, dist_dir]:
        if exists(rmdir):
            helpers.remove_file(rmdir)
    # Run the pyinstaller command (does all the work)
    _cmd('pyinstaller _setup/pyinstaller-hotspotter.spec')
    # Perform some post processing steps on the mac
    if sys.platform == 'darwin' and exists("dist/HotSpotter.app/Contents/"):
        copy_list = [
            'hsicon.icns',
            'Info.plist'
        ]
        srcdir = '_setup'
        dstdir = 'dist/HotSpotter.app/Contents/Resources/'
        for srcname, dstname in copy_list:
            src = join(srcdir, srcname)
            dst = join(dstdir, dstname)
            helpers.copy(src, dst)


def build_win32_inno_installer():
    inno_dir = r'C:\Program Files (x86)\Inno Setup 5'
    inno_fname = 'ISCC.exe'
    inno_fpath = join(inno_dir, inno_fname)
    iss_script = join('hotspotter', '_setup', 'wininstallerscript.iss')
    if not exists(inno_fpath):
        msg = '[setup] Inno not found and is needed for the win32 installer'
        print(msg)
        raise Exception(msg)
    args = [inno_fpath, iss_script]
    _cmd(args)


def compile_ui():
    'Compiles the qt designer *.ui files into python code'
    pyuic4_cmd = {'win32':  'C:\Python27\Lib\site-packages\PyQt4\pyuic4',
                  'linux2': 'pyuic4',
                  'darwin': 'pyuic4'}[sys.platform]
    widget_dir = join(dirname(realpath(__file__)), 'hotspotter/_frontend')
    print('[setup] Compiling qt designer files in %r' % widget_dir)
    for widget_ui in helpers.glob(widget_dir, '*.ui'):
        widget_py = os.path.splitext(widget_ui)[0] + '.py'
        cmd = ' '.join([pyuic4_cmd, '-x', widget_ui, '-o', widget_py])
        print('[setup] compile_ui()>' + cmd)
        os.system(cmd)


def clean():
    assert exists('setup.py'), 'must be run in hotspotter directory'
    assert exists('../hotspotter/setup.py'), 'must be run in hotspotter directory'
    assert exists('../hotspotter/hotspotter'), 'must be run in hotspotter directory'
    assert exists('_setup'), 'must be run in hotspotter directory'
    cwd = normpath(realpath(dirname(__file__)))
    print('[setup] Current working directory: %r' % cwd)
    helpers.remove_files_in_dir(cwd, '*.pyc', recursive=True)
    helpers.remove_files_in_dir(cwd, '*.prof', recursive=True)
    helpers.remove_files_in_dir(cwd, '*.prof.txt', recursive=True)
    helpers.remove_files_in_dir(cwd, '*.lprof', recursive=True)
    helpers.remove_files_in_dir(cwd, 'HotSpotterApp.*.pkg', recursive=False)
    helpers.remove_files_in_dir(cwd + '/hotspotter', '*.so', recursive=False)
    helpers.remove_files_in_dir(cwd + '/hotspotter', '*.c', recursive=False)
    helpers.delete(join(cwd, 'dist'))
    helpers.delete(join(cwd, 'build'))
    helpers.delete(join(cwd, "'"))  # idk where this file comes from


def fix_tpl_permissions():
    os.system('chmod +x hotspotter/_tpl/extern_feat/*.mac')
    os.system('chmod +x hotspotter/_tpl/extern_feat/*.ln')


def make_install_pyhesaff():
    dpath = normpath(HOME + '/code/hesaff')
    cmd = join(dpath, buildscript_fmt % 'hesaff')
    _cmd(cmd, sudo=True)


def make_install_pyflann():
    dpath = normpath(HOME + '/code/flann')
    cmd = join(dpath, buildscript_fmt % 'flann')
    _cmd(cmd, sudo=True)
    pass


def make_install_opencv():
    dpath = normpath(HOME + '/code/opencv')
    cmd = join(dpath, buildscript_fmt % 'opencv')
    _cmd(cmd, sudo=True)
    pass


def inrepo(func):
    def wrapper(repo, *args, **kwargs):
        repo_dpath = join(expanduser('~'), 'code', repo)
        cwd = os.getcwd()
        _cd(repo_dpath, False)
        result = func(repo, *args, **kwargs)
        _cd(cwd, False)
        print('')
        return result
    return wrapper


@inrepo
def pull(repo, branch=None):
    if repo == 'hotspotter':
        _cmd('git pull origin')
        _cmd('git pull github')
    else:
        _cmd('git pull')
    if branch is not None:
        _cmd('git checkout ' + branch)
        _cmd('git pull')


@inrepo
def push(repo):
    if repo == 'hotspotter':
        _cmd('git push origin')
        _cmd('git push github')
    else:
        _cmd('git push')


@inrepo
def status(repo):
    print('[helpers] ---- status(%r) ----' % repo)
    with helpers.Indenter('[%s]' % repo):
        _cmd('git status')


def compile_cython(fpath):
    pyinclude = '-I/usr/include/python2.7'
    gcc_flags = ' '.join(['-shared', '-pthread', '-fPIC', '-fwrapv', '-O2',
                          '-Wall', '-fno-strict-aliasing', pyinclude])
    fname, ext = splitext(fpath)
    fname_so = fname + '.so'
    fname_c  = fname + '.c'
    _cmd('cython ' + fpath)
    _cmd('gcc ' + gcc_flags + ' -o ' + fname_so + ' ' + fname_c)


def build_cython():
    compile_cython('hotspotter/spatial_verification2.py')
    compile_cython('hotspotter/matching_functions.py')


if __name__ == '__main__':
    print('[setup] Entering HotSpotter setup')
    for cmd in iter(sys.argv[1:]):
        if cmd in ['clean']:
            clean()
            sys.exit(0)
        if cmd in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)
        if cmd in ['installer', 'pyinstaller', 'build_pyinstaller', 'build_installer']:
            build_pyinstaller()
        if cmd in ['inno', 'win32inno']:
            build_win32_inno_installer()
        if cmd in ['flann', 'pyflann']:
            make_install_pyflann()
        if cmd in ['hesaff', 'pyhesaff']:
            make_install_pyhesaff()
        if cmd in ['opencv']:
            make_install_opencv()
        if cmd in ['pull']:
            pull('opencv')
            pull('hesaff')
            pull('flann')
            pull('hotspotter')
        if cmd in ['status']:
            status('opencv')
            status('hesaff')
            status('flann')
            status('hotspotter')
        if cmd in ['push']:
            #push('opencv')
            #push('hesaff')
            #push('flann')
            push('hotspotter')

        if cmd in ['update']:
            pull('opencv', 'hsbranch248')
            pull('hesaff', 'hotspotter_hesaff')
            pull('flann', 'hotspotter_flann')
            pull('hotspotter', 'jon')

        if cmd in ['cython']:
            build_cython()
