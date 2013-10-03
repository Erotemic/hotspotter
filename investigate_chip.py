#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import cv2
import spatial_verification2 as sv2
import sys
import params

from research import dump_groundtruth

def history_entry(database='', cx=-1, ocxs=[], cid=None):
    return (database, cx, ocxs)

HISTORY = [
    history_entry('GZ', 111, [305]),
    history_entry('TOADS', 32),
]

def top_matching_features(res, axnum=None, match_type=''):
    cx2_fs = res.cx2_fs_V
    cx_fx_fs_list = []
    for cx in xrange(len(cx2_fs)):
        fx2_fs = cx2_fs[cx]
        for fx in xrange(len(fx2_fs)):
            fs = fx2_fs[fx]
            cx_fx_fs_list.append((cx, fx, fs))

    cx_fx_fs_sorted = np.array(sorted(cx_fx_fs_list, key=lambda x: x[2])[::-1])

    sorted_score = cx_fx_fs_sorted[:,2]
    fig = df2.figure(0)
    df2.plot(sorted_score)

if __name__ == '__main__':
    if not 'hs' in vars():
        current = len(HISTORY) - 1
        (db, cx, ocx) = HISTORY[current]
        db_dir = eval('params.'+db)
        #
        hs = ld2.HotSpotter()
        hs.load_all(db_dir, matcher=False)
        hs.set_samples()

    dump_groundtruth.plot_name_cx(hs, cx)

    # Database descriptor + keypoints
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    def get_features(cx):
        rchip = hs.get_chip(cx)
        rchip_size = cx2_rchip_size[cx]
        fx2_kp   = cx2_kpts[cx]
        fx2_scale = sv2.keypoint_scale(fx2_kp)
        fx2_desc = cx2_desc[cx]
        return rchip, rchip_size, fx2_kp, fx2_scale, fx2_desc
    # Query features
    rchip1, rchip_size1, fx2_kp1, fx2_scale1, fx2_desc1 = get_features(qcx)
    # Result features
    rchip2, rchip_size2, fx2_kp2, fx2_scale2, fx2_desc2 = get_features(cx)
    # Vsmany index
    #c2.precompute_index_vsmany(hs)
    #qcx2_res = mc2.run_matching(hs)

    params.__MATCH_TYPE__ = 'vsmany'
    hs.load_matcher()
    vsmany_index = hs.matcher._Matcher__vsmany_index
    params.__VSMANY_SCORE_FN__ = 'LNRAT'
    resLNRAT = mc2.build_result_qcx(hs, qcx)
    fig1 = df2.show_match_analysis(hs, resLNRAT, N=5, fignum=1, figtitle=' LNRAT')
    #fig.tight_layout()

    params.__VSMANY_SCORE_FN__ = 'LNBNN'
    resLNBNN = mc2.build_result_qcx(hs, qcx)
    fig2 = df2.show_match_analysis(hs, resLNBNN, N=5, fignum=2, figtitle=' LNBNN')

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    resRATIO = mc2.build_result_qcx(hs, qcx)
    fig3 = df2.show_match_analysis(hs, resRATIO, N=5, fignum=3, figtitle=' RATIO')

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    
    params.__MATCH_TYPE__ = 'bagofwords'
    hs.load_matcher()
    resBOW = mc2.build_result_qcx(hs, qcx)
    fig4 = df2.show_match_analysis(hs, resBOW, N=5, fignum=4, figtitle=' bagofwords')
    
    params.__MATCH_TYPE__ = 'vsone'
    hs.load_matcher()
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    fig5 = df2.show_match_analysis(hs, res_vsone, N=5, fignum=5, figtitle=' vsone')


    fig6 = df2.show_match_analysis(hs, resLNBNN, N=20, fignum=6,
                                figtitle=' LNBNN More', show_query=False)

    df2.update()

    

    res = resLNBNN




exec(df2.present())
