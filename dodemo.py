#!/usr/bin/python2.7
from __init__ import *
from _demos import vsone

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    db_dir = params.GZ if len(sys.argv) == 1 else params.DEFAULT
    if not 'hs' in vars():
        hs = ld2.HotSpotter()
        hs.load_all(db_dir, matcher=False)
        qcx = helpers.get_arg_after('--qcx', type_=int)
        cx = helpers.get_arg_after('--cx', type_=int)
        if qcx is None:
            qcx = 1046
        if cx is None:
            cx_list = hs.get_other_cxs(qcx)
        else:
            cx_list = [cx]
        
    print('cx_list = %r ' % cx_list)
    for fignum, cx in enumerate(cx_list):
        vsone.show_vsone_demo(hs, qcx, cx, fignum=fignum)
    exec(df2.present())
'''
python demos.py vsone GZ --qcx 1046
'''
    
