# -*- mode: python -*-
# sudo python pyinstaller.py -F -w /hotspotter/main.py -i /hotspotter/hsicon.icns -n HotSpotter

import os
import sys 
from os.path import join, dirname, exists, normpath

#hsroot = '/hotspotter'
hsroot = os.getcwd()
if not exists(hsroot) or not exists(join(hsroot, 'setup.py')):
    raise Exception('You must run this script in the hotspotter root')

# This needs to be relative directory. Leave as is.
hsbuild = ''

# ------
# Build Analysis
main_py = join(hsroot, 'main.py')
scripts = [main_py]
a = Analysis(scripts,
             hiddenimports=[],
             hookspath=None)

# ------
# Specify Data in TOC format (SRC, DEST, TYPE)
#http://www.pyinstaller.org/export/develop/project/doc/Manual.html
#toc-class-table-of-contents
splash_rpath = '_frontend/splash.png'
splash_src   = join(hsroot,  splash_rpath)
splash_dest  = join(hsbuild, splash_rpath)
a.datas += [(splash_dest, splash_src, 'DATA')]

# Add TPL Libs for current platform
ROOT_DLLS = ['libgcc_s_dw2-1.dll', 'libstdc++-6.dll']

lib_rpath = normpath(join('_tpl/extern_feat/', sys.platform))
# Walk the lib dir
walk_path = join(hsroot, lib_rpath)
print "Adding lib files from directory:" , walk_path
for root, dirs, files in os.walk(walk_path):
    for lib_name in files:
        print "Adding lib name:", lib_name
        toc_src = join(hsroot, lib_rpath, lib_name)
        toc_dest = join(hsbuild, lib_rpath, lib_name)
        # MinGW libs should be put into root
        if lib_name in ROOT_DLLS: # or lib_name.find('libopencv') > -1:
            toc_dest = join(hsbuild, lib_name)
        print toc_dest, toc_src
        a.datas += [(toc_dest, toc_src, 'DATA')]

# Add documentation folder
a.datas += [('_doc/HotSpotterUserGuide.pdf', join(hsroot, '_doc','HotSpotterUserGuide.pdf'), 'DATA')]
# Add Landmark
# a.datas += [(hsroot, join(hsroot, hsroot), 'DATA')]

# Get Correct Icon
icon_cpmap = { 'darwin' : 'hsicon.icns',
               'win32'  : 'hsicon.ico' ,
               'linux2' : 'hsicon.ico' }
iconfile = join(hsroot, '_setup', icon_cpmap[sys.platform])

# Get Correct Extension
ext_cpmap  = {'darwin':'', 'win32':'.exe', 'linux2':'.ln'}
appext   = ext_cpmap[sys.platform]

if sys.platform == 'win32':
  exe_name = 'build/HotSpotterApp.exe'
elif sys.platform == 'linux2':
  exe_name = 'build/HotSpotterApp.ln'
elif sys.platform == 'darwin':
  exe_name = 'build/pyi.darwin/HotSpotterApp/HotSpotterApp'


pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=exe_name,
          debug=False,
          strip=None,
          upx=True,
          console = sys.platform != "darwin",
          icon=iconfile)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=join('dist', 'HotSpotter'))

bundle_name = 'HotSpotter'
if sys.platform == "darwin":
  bundle_name += '.app'

app = BUNDLE(coll, name=join('dist', bundle_name))
