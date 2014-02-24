from __future__ import division, print_function
from hsdev.dbgimport import *  # NOQA

exec(standard_imports)

if __name__ == 'main':
    multiprocessing.freeze_support()
    #exec(open('dbgimport.py').read())
    if '--img' in sys.argv:
        img_fpath_ = helpers.dbg_get_imgfpath()
        np_img_ = io.imread(img_fpath_)
        img_ = Image.open(img_fpath_)
