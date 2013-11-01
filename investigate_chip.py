#exec(open('__init__.py').read())
#exec(open('_research/investigate_chip.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import cv2
import spatial_verification2 as sv2
import helpers
import sys
import params
import vizualizations as viz
import voting_rules2 as vr2

def reload_module():
    import imp, sys
    print('[reload] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def history_entry(database='', cid=-1, ocids=[], notes='', cx=-1):
    return (database, cid, ocids, notes)

# A list of poster child examples. (curious query cases)
GZ_greater1_cid_list = [140, 297, 306, 311, 425, 441, 443, 444, 445, 450, 451,
                        453, 454, 456, 460, 463, 465, 501, 534, 550, 662, 786,
                        802, 838, 941, 981, 1043, 1046, 1047]
HISTORY = [
    history_entry('GZ', 1047,    [],               notes='extreme viewpoint #gt=4'),
    history_entry('GZ', 1046,    [],               notes='extreme viewpoint #gt=2'),
    history_entry('GZ', 786,     [787],            notes='foal #gt=11'),
    history_entry('GZ', 501,     [140],            notes='dark lighting'),
    history_entry('GZ', 941,     [900],            notes='viewpoint / quality'),
    history_entry('GZ', 981,     [802],            notes='foal extreme viewpoint'),
    history_entry('GZ', 838,     [801, 980],       notes='viewpoint / quality'),
    history_entry('GZ', 662,     [262],            notes='viewpoint / shadow (circle)'),
    history_entry('GZ', 311,     [289],            notes='quality'),
    history_entry('GZ', 297,     [301],            notes='quality'),
    history_entry('GZ', 306,     [112],            notes='occlusion'),
    history_entry('GZ', 534,     [411, 727],       notes='LNBNN failure'),
    history_entry('GZ', 463,     [173],            notes='LNBNN failure'),
    history_entry('GZ', 460,     [613, 460],       notes='background match'),
    history_entry('GZ', 465,     [589, 460],       notes='background match'),
    history_entry('GZ', 454,     [198, 447],       notes='forground match'),
    history_entry('GZ', 445,     [702, 435],       notes='forground match'),
    history_entry('GZ', 453,     [682, 453],       notes='forground match'),
    history_entry('GZ', 550,     [551, 452],       notes='forground match'),
    history_entry('GZ', 450,     [614],            notes='other zebra match'),
    history_entry('TOADS', cx=32),
    history_entry('MOTHERS', 69, [68],             notes='textured foal (lots of bad matches)'),
    history_entry('MOTHERS', 28, [27],             notes='viewpoint foal'),
    history_entry('MOTHERS', 53, [54],             notes='image quality'),
    history_entry('MOTHERS', 51, [50],             notes='dark lighting'),
    history_entry('MOTHERS', 44, [43, 45],         notes='viewpoint'),
    history_entry('MOTHERS', 66, [63, 62, 64, 65], notes='occluded foal'),
]


def quick_assign_vsmany(hs, qcx, cx, K): 
    #if hs.isindexed(qcx)
    K += 1
    desc1 = hs.feats.cx2_desc[qcx]
    vsmany_index = hs.matcher._Matcher__vsmany_index
    vsmany_flann = vsmany_index.vsmany_flann
    ax2_cx       = vsmany_index.ax2_cx
    ax2_fx       = vsmany_index.ax2_fx
    print('[invest] Quick vsmany over %s indexed descriptors. K=%r' %
          (helpers.commas(len(ax2_cx)), K))
    checks       = params.VSMANY_FLANN_PARAMS['checks']
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc1, K+1, checks=checks)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([mc2.LNBNN_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:, 0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:, 0:K]]
    # Build feature matches
    num_qf = len(desc1)
    qfx2_qfx = np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, K)) 
    iter_matches = iter(zip(qfx2_qfx.flat, qfx2_cx.flat,
                            qfx2_fx.flat, qfx2_score.flat))
    fm, fs = ([], [])
    for qfx, cx_, fx, score in iter_matches:
        if cx != cx_: continue
        fm.append((qfx, fx))
        fs.append(score)
    fm = mc2.fix_fm(fm)
    fs = mc2.fix_fs(fs)
    return fm, fs

def quick_assign_vsone(hs, qcx, cx, ratio_thresh=1.2, burst_thresh=None):
    print('[invest] Performing quick vsone')
    desc1 = hs.feats.cx2_desc[qcx]
    desc2 = hs.feats.cx2_desc[cx]
    vsone_flann, checks = mc2.get_vsone_flann(desc1)
    fm, fs = mc2.match_vsone(desc2, vsone_flann, checks, ratio_thresh, burst_thresh)
    return fm, fs

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


def build_voters_profile(hs, qcx, K):
    cx2_nx = hs.tables.cx2_nx
    hs.ensure_matcher_type('vsmany')
    K += 1
    desc1 = hs.feats.cx2_desc[qcx]
    vsmany_index = hs.matcher._Matcher__vsmany_index
    vsmany_flann = vsmany_index.vsmany_flann
    ax2_cx       = vsmany_index.ax2_cx
    ax2_fx       = vsmany_index.ax2_fx
    print('[invest] Building voter preferences over %s indexed descriptors. K=%r' %
          (helpers.commas(len(ax2_cx)), K))
    checks       = params.VSMANY_FLANN_PARAMS['checks']
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc1, K+1, checks=checks)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([mc2.LNBNN_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:, 0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:, 0:K]]
    qfx2_nx  = vr2.temporary_names(qfx2_cx, cx2_nx[qfx2_cx], zeroed_cx_list=[qcx])
    voters_profile = (qfx2_nx, qfx2_cx, qfx2_fx, qfx2_score)
    return voters_profile

def investigate_scoring_rules(hs, qcx, fnum=1):
    K = 4
    vr2.rrr()
    voters_profile = build_voters_profile(hs, qcx, K)
    vr2.apply_voting_rules(hs, qcx, voters_profile, fnum)
    fnum += 1
    return fnum


param1 = 'K'
param2 = 'xy_thresh'
assign_alg = 'vsmany'
nParam1=1 
fnum = 1
nParam2=1
cx_list='gt1'
def vary_query_params(hs, qcx, param1='ratio_thresh', param2='xy_thresh',
                      assign_alg='vsone', nParam1=3, nParam2=3, fnum=1,
                      cx_list='gt'):
    possible_variations = {
                        # start, #step  #props
        'K'            : (10,      20, 'int'),
        'ratio_thresh' : (1.6,   0.10),  
        'xy_thresh'    : (0.001, 0.01), 
        'scale_min'    : (0.5,   0.01),
        'scale_max'    : (2.0,  -0.01)
    }
    param_ranges = {
        'param1'    : [param1]+[list(possible_variations[param1])],
        'param2'    : [param2]+[list(possible_variations[param2])]
    }
    # Ground truth matches
    if cx_list == 'gt':
        cx_list = hs.get_groundtruth_cxs(qcx)
    if cx_list == 'gt1':
        gt_list = hs.get_groundtruth_cxs(qcx)
        cx_list = gt_list[0:1]
    #cx = cx_list[0]
    for cx in cx_list:
        fnum = vary_two_params(hs, qcx, cx, param_ranges, assign_alg,
                               nParam1, nParam2, fnum)
    return fnum


def linear_logspace(start, stop, num, base=2):
    return 2 ** np.linspace(np.log2(start), np.log2(stop), num)

def vary_two_params(hs, qcx, cx, param_ranges, assign_alg, nParam1=3, nParam2=3, fnum=1):
    # Query Features
    cx2_rchip_size = hs.get_cx2_rchip_size()
    get_features = quick_get_features_factory(hs)
    rchip1, fx2_kp1, fx2_desc1, cid1 = get_features(qcx)
    rchip2, fx2_kp2, fx2_desc2, cid2 = get_features(cx)
    rchip_size1 = cx2_rchip_size[qcx]
    rchip_size2 = cx2_rchip_size[cx]

    possible_assign_fns = {'vsone'   : quick_assign_vsone, 
                           'vsmany'  : quick_assign_vsmany, }
    #possible_space_fns = {'lin' : np.linspace,
                          #'log' : linear_logspace}
    quick_assign_fn = possible_assign_fns[assign_alg]

    # Varied Parameters
    def get_param(key, nParam):
        param = param_ranges[key][0]
        param_info = param_ranges[key][1]
        param_type = 'float' if len(param_info) <= 2 else param_info[2]
        #space_fn = possible_space_fns['lin' if len(param_info) <= 3 else param_info[3]]
        #param_range = list(param_info[0:2]) + [nParam]
        npnormal = np.random.normal
        start = param_info[0]
        step  = param_info[1]
        # 
        param_steps = [start]
        for ix in xrange(nParam-1):
            param_steps.append(param_steps[-1] + step)
        if param_type == 'int':
            param_steps = map(int, map(round, param_steps))
        return param, param_steps, nParam
    param1, param1_steps, nParam1 = get_param('param1', nParam1)
    param2, param2_steps, nParam2 = get_param('param2', nParam2)
    nRows = nParam1
    nCols = nParam2+1

    print('[invest] Varying parameters %r: nRows=%r, nCols=%r' % (assign_alg, nRows, nCols))
    print('[invest] %r = %r ' % (param1, param1_steps))
    print('[invest] %r = %r ' % (param2, param2_steps))
    # Assigned Features with param1
    for rowx, param1_value in enumerate(param1_steps):
        assign_args = {param1:param1_value}
        (fm, fs) = quick_assign_fn(hs, qcx, cx, **assign_args)
        def _show_matches_helper(fm, fs, rowx, colx, title):
            plotnum = (nRows, nCols, rowx*nCols+colx)
            #print('rowx=%r, colx=%r, plotnum=%r' % (rowx, colx, plotnum))
            df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs, fignum=fnum,
                            plotnum=plotnum, title=title, draw_pts=False)
        # Plot the original assigned matches
        title = param1+'=%.3e' % param1_value
        _show_matches_helper(fm, fs, rowx, 1, '')
        ax = df2.plt.gca()
        ylabel_args = dict(rotation='horizontal',
                           verticalalignment='bottom',
                           horizontalalignment='right')
        ax.set_ylabel(title, **ylabel_args)
        #if rowx == nRows - 1:
        def _set_xlabel(label):
            #if False or rowx == 0:
                #ax = df2.plt.gca()
                #ax.set_title(label)
            if rowx == nRows - 1:
                ax = df2.plt.gca()
                ax.set_xlabel(label)

        df2.adjust_subplots(left=0.05, right=1.0,
                            bottom=0.1, top=0.85,
                            wspace=0.01, hspace=0.01)
        _set_xlabel(assign_alg)
        # Spatially verify with params2
        for colx, param2_value in enumerate(param2_steps):
            sv_args = {'rchip_size2':rchip_size2, param2:param2_value}
            fm_V, fs_V = mc2.spatially_verify2(fx2_kp1, fx2_kp2, fm, fs, **sv_args)
            # Plot the spatially verified matches
            title = param2 + '=%.3e' % param2_value #helpers.commas(param2_value, 3)
            _show_matches_helper(fm_V, fs_V, rowx, colx+2, '')
            _set_xlabel(title)

    df2.set_figtitle(assign_alg+' vary '+param1+' and '+param2+' \n qcid=%r, cid=%r' % (cid1, cid2))
    fnum += 1
    return fnum

def quick_get_features_factory(hs):
    'builds a factory function'
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_cid = hs.tables.cx2_cid 
    def get_features(cx):
        rchip = hs.get_chip(cx)
        fx2_kp = cx2_kpts[cx]
        fx2_desc = cx2_desc[cx]
        cid = cx2_cid[cx]
        return rchip, fx2_kp, fx2_desc, cid
    return get_features

def show_vsone_matches(hs, qcx, fnum=1):
    hs.ensure_matcher_type('vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    return res_vsone, fnum

def where_did_vsone_matches_go(hs, qcx, fnum=1, K=100):
    '''Finds a set of vsone matches and a set of vsmany matches. 
    displays where the vsone matches are in the vsmany ranked lists'''
    # Ground truth matches
    gt_cxs = hs.get_other_cxs(qcx)
    # Query Features
    cx2_rchip_size = hs.get_cx2_rchip_size()
    get_features = quick_get_features_factory(hs)
    rchip1, qfx2_kp1, qfx2_desc1, qcid = get_features(qcx)
    # Get/Show vsone matches
    res_vsone, fnum = show_vsone_matches(hs, qcx, fnum)
    gt2_fm_V = res_vsone.cx2_fm_V[gt_cxs]
    # Get vsmany assigned matches (no spatial verification)
    hs.ensure_matcher_type('vsmany')
    vsmany_index = hs.matcher._Matcher__vsmany_index
    (qfx2_cx, qfx2_fx, qfx2_dists) = mc2.desc_nearest_neighbors(qfx2_desc1, vsmany_index, K)
    # Find where the matches to the correct images are
    print('[invest]  Finding where the vsone matches went for %s' % hs.vs_str(qcx, qcid))
    k_inds  = np.arange(0, K)
    qf_inds = np.arange(0, len(qfx2_cx))
    kxs, qfxs = np.meshgrid(k_inds, qf_inds)
    for gtx, ocx in enumerate(gt_cxs):
        rchip2, fx2_kp2, fx2_desc2, ocid = get_features(ocx)
        rchip_size2 = cx2_rchip_size[ocx]
        print('[invest] Checking matches to ground truth %r / %r %s' % 
              (gtx+1, len(gt_cxs), hs.vs_str(ocx, ocid)))
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
        fm_intersect, vsone_ix, vsmany_ix = helpers.intersect2d(vsone_fm_V, vsmany_fm)
        print(vsmany_ix)
        isecting_vsmany_fm = vsmany_fm[vsmany_ix]
        isecting_vsmany_fs = vsmany_kxs[vsmany_ix] # use k as score
        isecting_kxs = vsmany_kxs[vsmany_ix]
        # Spatially verify the vsmany matches 
        vsmany_fm_V, vsmany_fs_V = mc2.spatially_verify2(qfx2_kp1, fx2_kp2,
                                                         vsmany_fm, vsmany_fs,
                                                         rchip_size2=rchip_size2)
        # Intersect vsmany_V with vsone_V
        fm_V_intersect, vsoneV_ix, vsmanyV_ix = helpers.intersect2d(vsone_fm_V, vsmany_fm_V)
        isecting_vsmany_fm_V = vsmany_fm[vsmanyV_ix]
        print('[invest]   VSONE had %r verified matches to this image ' % (len(vsone_fm_V)))
        print('[invest]   In the top K=%r in this image...' % (K))
        print('[invest]   VSMANY had %r assignments to this image.' % (len(vsmany_qfxs)))
        print('[invest]   VSMANY had %r unique assignments to this image' % (len(np.unique(qfxs))))
        print('[invest]   VSMANY had %r verified assignments to this image' % (len(vsmany_fm_V)))
        print('[invest]   There were %r / %r intersecting matches in VSONE_V and VSMANY' % 
              (len(fm_intersect), len(vsone_fm_V)))
        print('[invest]   There were %r / %r intersecting verified matches in VSONE_V and VSMANY_V' % 
              (len(fm_V_intersect), len(vsone_fm_V)))
        print('[invest]   Distribution of kxs: '+helpers.printable_mystats(kxs))
        print('[invest]   Distribution of intersecting kxs: '+helpers.printable_mystats(isecting_kxs))
        # Visualize the intersecting matches 
        def _show_matches_helper(fm, fs, plotnum, title):
            df2.show_matches2(rchip1, rchip2, qfx2_kp1, fx2_kp2, fm, fs,
                              fignum=fnum, plotnum=plotnum, title=title, 
                              draw_pts=False)
        _show_matches_helper(vsmany_fm, vsmany_fs, (1,2,1), 'vsmany matches')
        #_show_matches_helper(vsmany_fm_V, vsmany_fs_V, (1,3,2), 'vsmany verified matches')
        _show_matches_helper(isecting_vsmany_fm, isecting_vsmany_fs, (1,2,2),
                             'intersecting vsmany K=%r matches' % (K,))
        df2.set_figtitle('vsmany K=%r qid%r vs cid%r'  % (K, qcid, ocid))
        # Hot colorscheme is black->red->yellow->white
        print('[invest] black->red->yellow->white')
        fnum+=1
    return fnum

def plot_name(hs, qcx, fnum=1):
    print('[invest] Plotting name')
    viz.plot_name_of_cx(hs, qcx, fignum=fnum)
    return fnum+1

def compare_matching_methods(hs, qcx, fnum=1):
    print('[invest] Comparing match methods')
    # VSMANY matcher
    hs.ensure_matcher_type('vsmany')
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
    hs.ensure_matcher_type('bagofwords')
    resBOW = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resBOW, N=5, fignum=fnum, figtitle=' bagofwords')
    fnum+=1
    # VSONE matcher
    hs.ensure_matcher_type('vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    # Extra 
    df2.show_match_analysis(hs, vsmany_results['LNBNN'], N=20, fignum=fnum,
                                figtitle=' LNBNN More', show_query=False)
    fnum+=1
    return fnum

def view_all_history_names_in_db(hs, db):
    qcid_list =[]
    for (db_, qcid, ocids, notes) in HISTORY:
        if db == db_:
            qcid_list += [qcid]
    print('qcid_list = %r ' % qcid_list)
    qcx_list = hs.cid2_cx(qcid_list)
    print('qcx_list = %r ' % qcid_list)
    nx_list = hs.tables.cx2_nx[qcx_list]
    unique_nxs = np.unique(nx_list)
    print('unique_nxs = %r' % unique_nxs)
    names = hs.tables.nx2_name[unique_nxs]
    print('names = %r' % names)
    helpers.ensuredir('hard_names')
    for nx in unique_nxs:
        viz.plot_name(hs, nx, fignum=nx)
        df2.save_figure(fpath='hard_names', usetitle=True)
    helpers.vd('hard_names')

def parse_arguments():
    '''
    Defines the arguments for investigate_chip.py
    '''
    print('==================')
    print('[invest] ---------')
    print('[invest] ARGPARSE')
    import argparse
    parser = argparse.ArgumentParser(description='HotSpotter - Investigate Chip', prefix_chars='+-')
    def_on  = {'action':'store_false', 'default':True}
    def_off = {'action':'store_true', 'default':False}
    addarg = parser.add_argument
    def add_meta(switch, type, default, help, nargs=1):
        dest = switch.strip('-').replace('-','_')
        addarg(switch, metavar=dest, type=type, default=default, help=help, nargs=nargs)
    def add_int(switch, default, help, **kwargs):
        add_meta(switch, int, default, help, **kwargs)
    def add_str(switch, default, help):
        add_meta(switch, str, default, help)
    def add_bool(switch, default, help):
        action = 'store_false' if default else 'store_true' 
        dest = switch.strip('-').replace('-','_')
        addarg(switch, dest=dest, action=action, default=default, help=help)
    def test_bool(switch):
        add_bool(switch, False, 'runs this test')
    add_int('--qcid',  1, 'query chip-id to investigate', nargs='*')
    add_int('--ocid',  [], 'query chip-id to investigate', nargs='*')
    add_int('--histid', None, 'history id (hard cases)')
    add_str('--db', 'MOTHERS', 'database to load')
    test_bool('--show-names')
    test_bool('--vary-vsmany-k-xythresh')
    test_bool('--vary-vsone-ratio-xythresh')
    args, unknown = parser.parse_known_args()
    print('[invest] args    = %r' % (args,))
    print('[invest] unknown = %r' % (unknown,))
    print('[invest] ---------')
    print('==================')
    return args

def run_investigations(qcx, args):
    print('[invest] Running Investigation: '+hs.cxstr(qcx))
    fnum = 1
    #view_all_history_names_in_db(hs, 'MOTHERS')
    if args.show_names:
        fnum = plot_name(hs, qcx, fnum)
    #fnum = compare_matching_methods(hs, qcx, fnum)
    if args.vary_vsone_ratio_xythresh:
        fnum = vary_query_params(hs, qcx, 'ratio_thresh', 'xy_thresh', 'vsone', 4, 4, fnum, cx_list='gt1')
    #hs.ensure_matcher_type('vsmany')
    #fnum = vary_query_params(hs, qcx, 'K', 'xy_thresh', 'vsmany', 4, 4, fnum, cx_list='gt1') #fnum = where_did_vsone_matches_go(hs, qcx, fnum, K=10)
    #fnum = where_did_vsone_matches_go(hs, qcx, fnum, K=20)
    #fnum = where_did_vsone_matches_go(hs, qcx, fnum, K=100)
    #fnum = investigate_scoring_rules(hs, qcx, fnum)

def hs_from_db(db):
    # Load hotspotter
    db_dir = eval('params.'+db)
    hs = ld2.HotSpotter()
    hs.load_all(db_dir, matcher=False)
    hs.set_samples()
    return hs

if __name__ == '__main__':
    #exec(open('investigate_chip.py').read())
    if not 'hs' in vars():
        args = parse_arguments()
        if not args.histid is None: # Grab an example
            (args.db, args.qcid, args.ocid, notes) = HISTORY[args.histid]
        hs = hs_from_db(args.db)
        qcxs = hs.cid2_cx(args.qcid)
        ocxs = hs.cid2_cx(args.ocid)
        if not np.iterable(qcxs):
            qcxs = [qcxs]
    print('[invest] running ')
    fmtstr = helpers.make_progress_fmt_str(len(qcxs), '[invest] investigation ')
    print('====================')
    for count, qcx in enumerate(qcxs):
        print(fmtstr % (count+1))
        run_investigations(qcx, args)
    print('====================')
    #df2.update()
exec(df2.present()) #**df2.OooScreen2()
