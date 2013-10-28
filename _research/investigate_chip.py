#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import cv2
import helpers
import spatial_verification2 as sv2
import sys
import params

from _research import dump_groundtruth

def history_entry(database='', cx=-1, ocxs=[], cid=None, notes=''):
    return (database, cx, ocxs, cid, notes)

# A list of poster child examples. (curious query cases)
HISTORY = [
    history_entry('TOADS', 32),
    history_entry('GZ', 111, [305]),
    history_entry('GZ', 1046, notes='viewpoint'),
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

def idontknowwhatiwasdoingwheniwrotethis():
    # Database descriptor + keypoints
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    def get_features(cx):
        rchip = hs.get_chip(cx)
        rchip_size = cx2_rchip_size[cx]
        fx2_kp     = cx2_kpts[cx]
        fx2_scale  = sv2.keypoint_scale(fx2_kp)
        fx2_desc   = cx2_desc[cx]
        return rchip, rchip_size, fx2_kp, fx2_scale, fx2_desc
    cx = ocxs[0]
    # Grab features
    rchip1, rchip_size1, fx2_kp1, fx2_scale1, fx2_desc1 = get_features(qcx)
    rchip2, rchip_size2, fx2_kp2, fx2_scale2, fx2_desc2 = get_features(cx)
    # Vsmany index
    #c2.precompute_index_vsmany(hs)
    #qcx2_res = mc2.run_matching(hs)

def intersect2d_numpy(A, B): 
    #http://stackoverflow.com/questions/8317022/
    #get-intersecting-rows-across-two-2d-numpy-arrays/8317155#8317155
    nrows, ncols = A.shape
    # HACK to get consistent dtypes
    assert A.dtype is B.dtype, 'A and B must have the same dtypes'
    dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
    try: 
        C = np.intersect1d(A.view(dtype), B.view(dtype))
    except ValueError as ex:
        C = np.intersect1d(A.copy().view(dtype), B.copy().view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C

def intersect2d(A, B):
    Cset  =  set(tuple(x) for x in A).intersection(set(tuple(x) for x in B))
    Ax = np.array([x for x, item in enumerate(A) if tuple(item) in Cset], dtype=np.int)
    Bx = np.array([x for x, item in enumerate(B) if tuple(item) in Cset], dtype=np.int)
    C = np.array(tuple(Cset))
    return C, Ax, Bx

def where_did_vsone_matches_go(hs, qcx, fnum=1):
    '''Finds a set of vsone matches and a set of vsmany matches. 
    displays where the vsone matches are in the vsmany ranked lists'''
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    def get_rchip_and_kpts(cx):
        rchip = hs.get_chip(cx)
        fx2_kp = cx2_kpts[cx]
        fx2_desc = cx2_desc[cx]
        cid = cx2_cid[cx]
        return rchip, fx2_kp, fx2_desc, cid
    cx2_cid = hs.tables.cx2_cid 
    rchip1, qfx2_kp1, qfx2_desc1, qcid = get_rchip_and_kpts(qcx)
    set_matcher_type(hs, 'vsone')
    # Get vsone matches
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    gt_cxs = hs.get_other_cxs(qcx)
    gt2_fm_V = res_vsone.cx2_fm_V[gt_cxs]

    # Get vsmany assigned matches (no spatial verification)
    set_matcher_type(hs, 'vsmany')
    vsmany_index = hs.matcher._Matcher__vsmany_index
    K = 100
    (qfx2_cx, qfx2_fx, qfx2_dists) = mc2.desc_nearest_neighbors(qfx2_desc1, vsmany_index, K)
    # Find where the matches to the correct images are
    print('Finding where the vsone matches went for qcx=%r, qcid=%r' % (qcx, qcid))
    k_inds  = np.arange(0, K)
    qf_inds = np.arange(0, len(qfx2_cx))
    kxs, qfxs = np.meshgrid(k_inds, qf_inds)
    for gtx, ocx in enumerate(gt_cxs):
        rchip2, fx2_kp2, fx2_desc2, ocid = get_rchip_and_kpts(ocx)
        rchip2_size = cx2_rchip_size[ocx]
        print('Checking matches to ground truth %r / %r cx=%r, cid=%r' % 
              (gtx+1, len(gt_cxs), ocx, ocid))
        # Get vsone indexes
        vsone_fm_V = gt2_fm_V[gtx]
        # Find correct feature and rank indexes: fx and kx
        vsmany_qfxs, vsmany_kxs = np.where(qfx2_cx == ocx)
        # Get comparisons to vsone
        qfx_kx_tup = zip(vsmany_qfxs, vsmany_kxs)
        vsmany_fxs = np.array([qfx2_fx[qfx, kx] for qfx, kx in qfx_kx_tup])
        def cast_uint32(arr):
            return np.array(arr, dtype=np.uint32)
        vsmany_fm  = cast_uint32(np.vstack(map(cast_uint32,(vsmany_qfxs, vsmany_fxs))).T)
        vsmany_fs  = vsmany_kxs # use k as score
        # Intersect vsmany with vsone_V
        fm_intersect, vsone_ix, vsmany_ix = intersect2d(vsone_fm_V, vsmany_fm)
        print(vsmany_ix)
        isecting_vsmany_fm = vsmany_fm[vsmany_ix]
        isecting_vsmany_fs = vsmany_kxs[vsmany_ix] # use k as score
        isecting_kxs = vsmany_kxs[vsmany_ix]
        # Spatially verify the vsmany matches 
        vsmany_fm_V, vsmany_fs_V, _2 = mc2.spatially_verify(qfx2_kp1, fx2_kp2, rchip2_size, vsmany_fm, vsmany_fs, qcx, ocx)
        # Intersect vsmany_V with vsone_V
        fm_V_intersect, vsoneV_ix, vsmanyV_ix = intersect2d(vsone_fm_V, vsmany_fm_V)
        isecting_vsmany_fm_V = vsmany_fm[vsmanyV_ix]
        print('  VSONE had %r verified matches to this image ' % (len(vsone_fm_V)))
        print('  In the top K=%r in this image...' % (K))
        print('  VSMANY had %r assignments to this image.' % (len(vsmany_qfxs)))
        print('  VSMANY had %r unique assignments to this image' % (len(np.unique(qfxs))))
        print('  VSMANY had %r verified assignments to this image' % (len(vsmany_fm_V)))
        print('  There were %r / %r intersecting matches in VSONE_V and VSMANY' % 
              (len(fm_intersect), len(vsone_fm_V)))
        print('  There were %r / %r intersecting verified matches in VSONE_V and VSMANY_V' % 
              (len(fm_V_intersect), len(vsone_fm_V)))
        print('  Distribution of kxs: '+helpers.printable_mystats(kxs))
        print('  Distribution of intersecting kxs: '+helpers.printable_mystats(isecting_kxs))
        # Visualize the intersecting matches 
        def plot_helpers(fm, fs, plotnum, title):
            df2.show_matches2(rchip1, rchip2, qfx2_kp1, fx2_kp2, fm, fs,
                              fignum=fnum, plotnum=plotnum, title=title, 
                              draw_pts=False)
        plot_helpers(vsmany_fm, vsmany_fs, (1,3,1), 'vsmany matches')
        plot_helpers(vsmany_fm_V, vsmany_fs_V, (1,3,2), 'vsmany verified matches')
        plot_helpers(isecting_vsmany_fm, isecting_vsmany_fs, (1,3,3), 'intersecting vsmany matches')
        df2.set_figtitle('vsmany qid%r vs cid%r'  % (qcid, ocid))
        # Hot colorscheme is black->red->yellow->white
        print('black->red->yellow->white')
        fnum+=1

        

    return fnum

def set_matcher_type(hs, match_type):
    print('Setting matcher type to: '+str(match_type))
    params.__MATCH_TYPE__ = match_type
    hs.load_matcher()


def plot_name(hs, qcx, fnum=1):
    print('Plotting name')
    dump_groundtruth.plot_name_cx(hs, qcx, fignum=fnum)
    return fnum+1


def compare_matching_methods(hs, qcx, fnum=1):
    print('Comparing match methods')
    # VSMANY matcher
    set_matcher_type(hs, 'vsmany')
    vsmany_score_options = ['LNRAT', 'LNBNN', 'RATIO']
    vsmany_index = hs.matcher._Matcher__vsmany_index
    vsmany_results = {}
    for score_type in vsmany_score_options:
        params.__VSMANY_SCORE_FN__ = score_type
        res_vsmany = mc2.build_result_qcx(hs, qcx)
        df2.show_match_analysis(hs, res_vsmany, N=5, fignum=fnum, figtitle=' LNRAT')
        vsmany_results[score_type] = res_vsmany
        fnum+=1
    # BAGOFWORDS matcher
    set_matcher_type(hs, 'bagofwords')
    resBOW = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resBOW, N=5, fignum=fnum, figtitle=' bagofwords')
    fnum+=1
    # VSONE matcher
    set_matcher_type(hs, 'vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    # Extra 
    df2.show_match_analysis(hs, vsmany_results['LNBNN'], N=20, fignum=fnum,
                                figtitle=' LNBNN More', show_query=False)
    fnum+=1
    return fnum

if __name__ == '__main__':
    if not 'hs' in vars():
        # Grab an example
        current = len(HISTORY) - 1
        (db, qcx, ocxs, cid, notes) = HISTORY[current]
        db_dir = eval('params.'+db)
        # Load hotspotter
        hs = ld2.HotSpotter()
        hs.load_all(db_dir, matcher=False)
        hs.set_samples()

    fnum = 1

    #fnum = plot_name(hs, qcx, fnum)
    #fnum = compare_matching_methods(hs, qcx, fnum)
    fnum = where_did_vsone_matches_go(hs, qcx, fnum)

    df2.update()

exec(df2.present())
