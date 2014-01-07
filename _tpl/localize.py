from os.path import expanduser, join
from hotspotter import helpers

# Localize hessian affine code
code_dir   = join(expanduser('~'), 'code')
hsdir      = join(code_dir, 'hotspotter')
extern_dir = join(hsdir, '_tpl', 'extern_feat')
hesaffsrc_dir = join(code_dir, 'hesaff')

hesaffbuild_dir = join(hesaffsrc_dir, 'build')
filemap = {
    hesaffbuild_dir: ['hesaffexe.exe',
                      'libhesaff.so',
                      'libhesaff.dylib',
                      'hesaffexe.ln',
                      'libhesaff.dll'],
    hesaffsrc_dir: ['pyhesaff.py'],
}

for srcdir, fname_list in filemap.iteritems():
    for fname in fname_list:
        src  = join(srcdir, fname)
        dest = join(extern_dir, fname)
        try:
            helpers.copy(src, dest)
        except Exception as ex:
            print(ex)

#raw_input('[_tpl/localize] Press enter to continue')
