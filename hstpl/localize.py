
import sys
from os.path import expanduser, join, exists

# Localize hessian affine code
code_dir   = join(expanduser('~'), 'code')
hsdir      = join(code_dir, 'hotspotter')
if not exists(hsdir):
    # For advisors computer
    code_dir = join(expanduser('~'), 'Code-RPI')
    hsdir      = join(code_dir, 'hotspotter')
    if not exists(hsdir):
        print(('[pyhesaff] hsdir=%r DOES NOT EXIST!' % (hsdir,)))
        raise Exception('Expected that hesaff and hotspotter to be in ~/code')

# Ensure hotspotter is in path before importing it
if not hsdir in sys.path:
    # Append hotspotter dir to PYTHON_PATH (i.e. sys.path)
    sys.path.append(hsdir)

from hscom import helpers
from hscom import helpers as util

extern_dir = join(hsdir, 'hstpl', 'extern_feat')
hesaffsrc_dir = join(code_dir, 'hesaff')

hesaffbuild_dir = join(hesaffsrc_dir, 'build')

built_files = {
    'linux2': ['hesaffexe', 'hesaffexe.ln', 'libhesaff.so'],
    'win32':  ['hesaffexe.exe', 'libhesaff.dll'],
    'darwin': ['hesaffexe', 'hesaffexe.mac', 'libhesaff.dylib']}[sys.platform]

filemap = {
    hesaffbuild_dir: built_files,
    hesaffsrc_dir: ['pyhesaff.py',
                    'ellipse.py',
                    'pyhesaffexe.py',
                    'ctypes_interface.py'], }

for srcdir, fname_list in list(filemap.items()):
    for fname in fname_list:
        src  = join(srcdir, fname)
        dest = join(extern_dir, fname)
        try:
            helpers.copy(src, dest)
        except Exception as ex:
            print(ex)

#raw_input('[_tpl/localize] Press enter to continue')
