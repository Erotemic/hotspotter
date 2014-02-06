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
APPLE = PLATFORM == 'darwin'
WIN32 = PLATFORM == 'win32'
LINUX = PLATFORM == 'linux2'

LIB_EXT = {'win32': '.dll',
           'darwin': '.dylib',
           'linux2': '.so'}[PLATFORM]


def join_SITE_PACKAGES(*args):
    import site
    from os.path import join, exists
    tried_list = []
    for dir_ in site.getsitepackages():
        path = join(dir_, *args)
        tried_list.append(path)
        if exists(path):
            return path
    msg = ('Cannot find: join_SITE_PACKAGES(*%r)\n'  % (args,))
    msg += 'Tried: \n    ' + '\n    '.join(tried_list)
    print(msg)
    raise Exception(msg)

# run from root
root_dir = os.getcwd()
try:
    assert exists(join(root_dir, 'setup.py'))
    assert exists('../hotspotter')
    assert exists('../hotspotter/hotspotter')
    assert exists('../hotspotter/setup.py')
    assert exists(root_dir)
except AssertionError:
    raise Exception('setup.py must be run from hotspotter root')


def add_data(a, dst, src):
    import textwrap
    from hscom import helpers
    from os.path import dirname, normpath, splitext
    global LIB_EXT

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
        if sys.platform == 'win32':
            path2 = fixwin32_shortname(path1)
        else:
            path2 = path1
        return path2

    src = platform_path(src)
    dst = dst
    helpers.ensurepath(dirname(dst))
    pretty_path = lambda str_: str_.replace('\\', '/')
    # Default datatype is DATA
    dtype = 'DATA'
    # Infer datatype from extension
    extension = splitext(dst)[1].lower()
    if extension == LIB_EXT.lower():
        dtype = 'BINARY'
    print(textwrap.dedent('''
    [setup] a.add_data(
    [setup]    dst=%r,
    [setup]    src=%r,
    [setup]    dtype=%s)''').strip('\n') %
          (pretty_path(dst), pretty_path(src), dtype))
    a.datas.append((dst, src, dtype))


# This needs to be relative to build directory. Leave as is.
hsbuild = ''

# ------
# Build Analysis
main_py = join(root_dir, 'main.py')
scripts = [main_py]
a = Analysis(scripts, hiddenimports=[], hookspath=None)  # NOQA

# ------
# Specify Data in TOC (table of contents) format (SRC, DEST, TYPE)
splash_src   = join(root_dir, 'hsgui/_frontend/splash.png')
splash_dst  = join(hsbuild, 'hsgui/_frontend/splash.png')
add_data(a, splash_dst, splash_src)

src = join(root_dir, 'hsgui/_frontend/hsicon.ico')
dst = join(hsbuild, 'hsgui/_frontend/hsicon.ico')
add_data(a, dst, src)

src = join(root_dir, 'hsgui/_frontend/resources_MainSkel.qrc')
dst = join(hsbuild, 'hsgui/_frontend/resources_MainSkel.qrc')
add_data(a, dst, src)

# Add TPL Libs for current PLATFORM
ROOT_DLLS = ['libgcc_s_dw2-1.dll', 'libstdc++-6.dll']


#/usr/local/lib/python2.7/dist-packages/pyflann/lib/libflann.so
# FLANN Library
libflann_fname = 'libflann' + LIB_EXT
if WIN32:
    libflann_src = join_SITE_PACKAGES('pyflann', 'lib', libflann_fname)
    libflann_dst = join(hsbuild, libflann_fname)
    add_data(a, libflann_dst, libflann_src)

if APPLE:
    try:
        libflann_src = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pyflann/lib/libflann.dylib'
        libflann_dst = join(hsbuild, libflann_fname)
        add_data(a, libflann_dst, libflann_src)
    except Exception as ex:
        print(repr(ex))

    libhesaff_fname = 'libhesaff' + LIB_EXT
    libhesaff_src = join(root_dir, 'hstpl', 'extern_feat', libhesaff_fname)
    libhesaff_dst = join(hsbuild, 'hstpl', 'extern_feat', libhesaff_fname)
    add_data(a, libhesaff_dst, libhesaff_src)

    # We need to add these 4 opencv libraries because pyinstaller does not find them.
    missing_cv_name_list = [
        'libopencv_videostab.2.4',
        'libopencv_superres.2.4',
        'libopencv_stitching.2.4',
    ]
    for name in missing_cv_name_list:
        fname = name + LIB_EXT
        src = join('/opt/local/lib', fname)
        dst = join(hsbuild, fname)
        add_data(a, dst, src)

lib_rpath = join('hstpl', 'extern_feat')

# Local dynamic Libraries
walk_path = join(root_dir, lib_rpath)
for root, dirs, files in os.walk(walk_path):
    for lib_fname in files:
        if fnmatch.fnmatch(lib_fname, '*' + LIB_EXT):
            # tpl libs should be relative to hotspotter
            toc_src  = join(root_dir, lib_rpath, lib_fname)
            toc_dst = join(hsbuild, lib_rpath, lib_fname)
            # MinGW libs should be put into root
            if lib_fname in ROOT_DLLS:
                toc_dst = join(hsbuild, lib_fname)
            add_data(a, toc_dst, toc_src)

# Qt GUI Libraries
walk_path = '/opt/local/Library/Frameworks/QtGui.framework/Versions/4/Resources/qt_menu.nib'
for root, dirs, files in os.walk(walk_path):
    for lib_fname in files:
        toc_src = join(walk_path, lib_fname)
        toc_dst = join('qt_menu.nib', lib_fname)
        add_data(a, toc_dst, toc_src)

# Documentation
userguide_dst = '_doc/HotSpotterUserGuide.pdf'
userguide_src = join(root_dir, '_doc/HotSpotterUserGuide.pdf')
add_data(a, userguide_dst, userguide_src)

# Icon File
ICON_EXT = {'darwin': 'icns',
            'win32':  'ico',
            'linux2': 'ico'}[PLATFORM]
iconfile = join(root_dir, '_setup', 'hsicon.' + ICON_EXT)

# Executable name
exe_name = {'win32':  'build/HotSpotterApp.exe',
            'darwin': 'build/pyi.darwin/HotSpotterApp/HotSpotterApp',
            'linux2': 'build/HotSpotterApp.ln'}[PLATFORM]

# TODO:
# http://www.pyinstaller.org/export/develop/project/doc/Manual.html?format=raw#accessing-data-files

# http://www.pyinstaller.org/export/develop/project/doc/Manual.html?format=raw#exe
pyz = PYZ(a.pure)   # NOQA

exe_kwargs = {
    'console': True,
    'debug': False,
    'name': exe_name,
    'exclude_binaries': True,
    'append_pkg': False,
}

collect_kwargs = {
    'strip': None,
    'upx': True,
    'name': join('dist', 'hotspotter')
}

# PYINSTALLER DOC:
# Only the following command-line options have an effect when
# building from a spec file:
# --upx-dir=
# --distpath=
# --workpath=
# --noconfirm
# --ascii

#exe_kwargs['upx']   = True
#exe_kwargs['onedir'] = True
#exe_kwargs['onefile'] = False
#exe_kwargs['windowed'] = False

# Windows only EXE options
if WIN32:
    exe_kwargs['icon'] = iconfile
    #exe_kwargs['version'] = 1.5


if APPLE:
    exe_kwargs['console'] = False

# Pyinstaller will gather .pyos
opt_flags = [('O', '', 'OPTION')]
exe = EXE(pyz, a.scripts + opt_flags, **exe_kwargs)   # NOQA

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, **collect_kwargs)  # NOQA

bundle_name = 'HotSpotter'
if APPLE:
    bundle_name += '.app'

app = BUNDLE(coll, name=join('dist', bundle_name))  # NOQA
