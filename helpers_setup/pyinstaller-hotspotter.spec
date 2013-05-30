# -*- mode: python -*-
# sudo python pyinstaller.py -F -w /hotspotter/main.py -i /hotspotter/hsicon.icns -n HotSpotter

import os
import sys 
from os.path import join, dirname, exists
pathex = dirname(__file__)
hotspotter_mainpy = None



#hsroot = '/hotspotter'
hsroot = os.getcwd()
if not exists(hsroot) or not exists(join(hsroot, 'setup.py')):
    raise Exception('You must run this script in the hotspotter root')

main_py = join(hsroot, 'main.py')
lib_tpl = join('hotspotter/tpl/lib/', sys.platform)

scripts = [main_py]

a = Analysis(scripts,
             hiddenimports=[],
             hookspath=None)


# Specify Data Tuples (Source, Dest, DataTypes)
splash_src = 'hotspotter/front/splash.png'
splash_dest = join(hsroot,splash_src)
a.datas += [(splash_src, splash_dest, 'DATA')]

lib_tpl_src = join(hsroot, lib_tpl)
for root, dirs, files in os.walk(lib_tpl_src):
    for lib_name in files:
        lib_src = join(lib_tpl, lib_name)
        lib_dest = join(hsroot, lib_src)
        a.datas += [(lib_src, lib_dest, 'BINARY')]
  

#
if sys.platform == 'dawrin':
    iconfile = join(hsroot, 'hsicon.icns')
else:
    iconfile = join(hsroot, 'hsicon.ico')

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/HotSpotterApp', 'HotSpotterApp'),
          debug=False,
          strip=None,
          upx=True,
          console=False,
          icon='/hotspotter/hsicon.icns')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=os.path.join('dist', 'HotSpotterApp'))

app = BUNDLE(coll,
             name=os.path.join('dist', 'HotSpotter.app'))
