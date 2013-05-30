# -*- mode: python -*-
# sudo python pyinstaller.py -F -w /hotspotter/main.py -i /hotspotter/hsicon.icns -n HotSpotter
a = Analysis(['/hotspotter/main.py'],
             pathex=['/Users/bluemellophone/Downloads/pyinstaller-2.0'],
             hiddenimports=[],
             hookspath=None)

a.datas += [('hotspotter/front/splash.png', '/hotspotter/hotspotter/front/splash.png',  'DATA')]

for root, dirs, files in os.walk('/hotspotter/hotspotter/tpl/lib/darwin/'):
  for file_name in files:
    a.datas += [('hotspotter/tpl/lib/darwin/' + file_name, '/hotspotter/hotspotter/tpl/lib/darwin/' + file_name,  'DATA')]
  

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/HotSpotter2', 'HotSpotter2'),
          debug=False,
          strip=None,
          upx=True,
          console=False , icon='/hotspotter/hsicon.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=os.path.join('dist', 'HotSpotter2'))
app = BUNDLE(coll,
             name=os.path.join('dist', 'HotSpotter2.app'))