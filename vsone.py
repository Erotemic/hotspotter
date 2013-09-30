from __init__ import *


# Database descriptor + keypoints
def get_features(cx):
    rchip    = hs.get_chip(cx)
    fx2_kp   = hs.feats.cx2_kpts[cx]
    fx2_desc = hs.feats.cx2_desc[cx]
    return rchip, fx2_kp, fx2_desc

def get_vsone_data(query_feats, result_feats):
    ' Assigned matches (vsone)'
    rchip1, fx2_kp1, fx2_desc1 = query_feats
    rchip2, fx2_kp2, fx2_desc2 = result_feats
    rchip_size2 = rchip2.size
    vsone_flann, checks = mc2.get_vsone_flann(fx2_desc1)
    fm, fs              = mc2.match_vsone(fx2_desc2, vsone_flann, checks)
    fm_V, fs_V, H       = mc2.spatially_verify(fx2_kp1, fx2_kp2, rchip_size2, fm, fs, qcx, cx)
    score = fs.sum(); score_V = fs_V.sum()
    vsone_cx_assign = fm, fs, score
    vsone_cx_svout = fm_V, fs_V, score_V
    return vsone_cx_assign, vsone_cx_svout

def show_vsone_data(query_feats, result_feats, vsone_data, fignum=0, figtitle=''):
    vsone_cx_assign, vsone_cx_svout = vsone_data
    # Show vsone
    rchip1, fx2_kp1, fx2_desc1 = query_feats
    rchip2, fx2_kp2, fx2_desc2 = result_feats
    fm, fs, score       = vsone_cx_assign
    fm_V, fs_V, score_V = vsone_cx_svout
    plot_kwargs = dict(all_kpts=False, fignum=fignum)
    df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs,
                        plotnum=(1,2,1), title='vsone assign', **plot_kwargs)
    df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm_V, fs_V,
                        plotnum=(1,2,2), title='vsone verified', **plot_kwargs)
    df2.set_figtitle(figtitle)

# ------------------

def show_vsone_demo(qcx, cx, fignum=0):
    print('[demo] vsone')
    print('qcx=%r, cx=%r' % (qcx, cx))
    # Query and Result features
    query_feats = get_features(qcx)
    result_feats = get_features(cx)
    query_uid = params.get_query_uid()
    vsone_data = get_vsone_data(query_feats, result_feats)
    figtitle = '%r v %r -- vsone' % (qcx, cx)
    show_vsone_data(query_feats, result_feats, vsone_data, fignum, figtitle)

# Assign + Verify
#params.__USE_CHIP_EXTENT__ = False
#vsone_data = get_vsone_data(cx)
#params.__USE_CHIP_EXTENT__ = True
#vsone_data2 = get_vsone_data(cx)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    db_dir = params.GZ if sys.argv == 1 else params.DEFAULT
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
        show_vsone_demo(qcx, cx, fignum=fignum)
    exec(df2.present())
'''
python vsone.py GZ --qcx 1046
'''
