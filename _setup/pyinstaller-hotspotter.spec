# -*- mode: python -*-
# sudo python pyinstaller.py -F -w /hotspotter/main.py -i /hotspotter/_setup/hsicon.icns -n HotSpotter
#http://www.pyinstaller.org/export/develop/project/doc/Manual.html
#toc-class-table-of-contents
import os
import sys
from os.path import join, exists
import fnmatch

# System Variables
PLATFORM = sys.platform


def join_SITE_PACKAGES(*args):
    import site
    from os.path import join, exists
    for dir_ in site.getsitepackages():
        path = join(dir_, *args)
        if exists(path):
            return path
    raise Exception('cannot find: %r' % (args,))

# run from root
root_dir = os.getcwd()
try:
    assert exists(join(root_dir, 'setup.py'))
    assert exists('../hotspotter')
    assert exists('../hotspotter/setup.py')
    assert exists(root_dir)
except AssertionError:
    raise Exception('Setup.py must be run from hotspotter/')

# This needs to be relative to build directory. Leave as is.
hsbuild = ''

# ------
# Build Analysis
main_py = join(root_dir, 'main.py')
scripts = [main_py]
a = Analysis(scripts, hiddenimports=[], hookspath=None)  # NOQA


def add_data(a, dst, src):
    import textwrap
    from hotspotter import helpers
    from os.path import dirname, normpath

    def fixwin32_shortname(path1):
        import ctypes
        try:
            #import win32file
            #buf = ctypes.create_unicode_buffer(buflen)
            path1 = unicode(path1)
            buflen = 260  # max size
            buf = ctypes.create_unicode_buffer(buflen)
            ctypes.windll.kernel32.GetLongPathNameW(path1, buf, buflen)
            #win32file.GetLongPathName(path1, )
            path2 = buf.value
        except Exception as ex:
            path2 = path1
            print(ex)
        return path2

    def platform_path(path):
        path1 = normpath(path)
        path2 = fixwin32_shortname(path1)
        return path2
    src = platform_path(src)
    dst = dst
    helpers.ensurepath(dirname(dst))
    pretty_path = lambda str_: str_.replace('\\', '/')
    print(textwrap.dedent('''
    [setup] a.add_data(
    [setup]    dst=%r,
    [setup]    src=%r)''').strip('\n') % tuple(map(pretty_path, (dst, src))))
    a.datas.append((dst, src, 'DATA'))

# ------
# Specify Data in TOC format (SRC, DEST, TYPE)
splash_src   = join(root_dir, '_frontend/splash.png')
splash_dst  = join(hsbuild, '_frontend/splash.png')
add_data(a, splash_dst, splash_src)

src = join(root_dir, '_frontend/', 'hsicon.ico')
dst = join(hsbuild, '_frontend/', 'hsicon.ico')
add_data(a, dst, src)

src = join(root_dir, '_frontend/' 'resources_MainSkel.qrc')
dst = join(hsbuild, '_frontend/' 'resources_MainSkel.qrc')
add_data(a, dst, src)

# Add TPL Libs for current PLATFORM
ROOT_DLLS = ['libgcc_s_dw2-1.dll', 'libstdc++-6.dll']

LIB_EXT = {'win32': 'dll',
           'darwin': 'dylib',
           'linux2': 'so'}[PLATFORM]

#/usr/local/lib/python2.7/dist-packages/pyflann/lib/libflann.so
# FLANN Library
libflann_fname = 'libflann.' + LIB_EXT
libflann_src = join_SITE_PACKAGES('pyflann', 'lib', libflann_fname)
libflann_dst = join(hsbuild, libflann_fname)
add_data(a, libflann_dst, libflann_src)


lib_rpath = join('_tpl', 'extern_feat')

# Local dynamic Libraries
walk_path = join(root_dir, lib_rpath)
for root, dirs, files in os.walk(walk_path):
    for lib_fname in files:
        if fnmatch.fnmatch(lib_fname, '*.' + LIB_EXT):
            # tpl libs should be relative to hotspotter
            toc_src  = join(root_dir, lib_rpath, lib_fname)
            toc_dst = join(hsbuild, lib_rpath, lib_fname)
            # MinGW libs should be put into root
            if lib_fname in ROOT_DLLS:
                toc_dst = join(hsbuild, lib_fname)
            add_data(a, toc_dst, toc_src)

# Documentation
userguide_dst = '_doc/HotSpotterUserGuide.pdf'
userguide_src = join(root_dir, '_doc', 'HotSpotterUserGuide.pdf')
add_data(a, userguide_dst, userguide_src)

# Icon File
ICON_EXT = {'darwin': 'icns',
            'win32':  'ico',
            'linux2': 'ico'}[PLATFORM]
iconfile = join(root_dir, '_setup', 'hsicon.' + ICON_EXT)

# Executable name
exe_name = {'win32':   'build/HotSpotterApp.exe',
            'darwin': 'build/pyi.darwin/HotSpotterApp/HotSpotterApp',
            'linux2': 'build/HotSpotterApp.ln'}[PLATFORM]

pyz = PYZ(a.pure)   # NOQA
exe_kwargs = dict(exclude_binaries=True, name=exe_name,
                  debug=False, strip=None,
                  upx=True, console=True,
                  #onefile='HotSpotterApp',
                  icon=iconfile)
                  #console = PLATFORM != 'darwin',
exe = EXE(pyz, a.scripts, **exe_kwargs)   # NOQA

collect_kwargs = dict(strip=None, upx=True, name=join('dist', 'hotspotter'))
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, **collect_kwargs)  # NOQA

bundle_name = 'hotspotter'
if PLATFORM == 'darwin':
    bundle_name += '.app'

app = BUNDLE(coll, name=join('dist', bundle_name))  # NOQA
