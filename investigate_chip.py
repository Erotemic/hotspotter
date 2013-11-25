#exec(open('__init__.py').read())
#exec(open('_research/investigate_chip.py').read())
from __future__ import division, print_function
import __builtin__
import argparse
import sys
from os.path import join

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[invest] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()

# Moved this up for faster help responce time
def parse_arguments():
    '''
    Defines the arguments for investigate_chip.py
    '''
    print('==================')
    print('[invest] ---------')
    print('[invest] ARGPARSE')
    parser = argparse.ArgumentParser(description='HotSpotter - Investigate Chip', prefix_chars='+-')
    def_on  = {'action':'store_false', 'default':True}
    def_off = {'action':'store_true', 'default':False}
    addarg = parser.add_argument
    def add_meta(switch, type, default, help, step=None, **kwargs):
        dest = switch.strip('-').replace('-','_')
        addarg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)
        if not step is None:
            add_meta(switch+'-step', type, step, help='', **kwargs)
    def add_bool(switch, default=True, help=''):
        action = 'store_false' if default else 'store_true' 
        dest = switch.strip('-').replace('-','_')
        addarg(switch, dest=dest, action=action, default=default, help=help)
    def add_var(switch, default, help, **kwargs):
        add_meta(switch, None, default, help, **kwargs)
    def add_int(switch, default, help, **kwargs):
        add_meta(switch, int, default, help, **kwargs)
    def add_float(switch, default, help, **kwargs):
        add_meta(switch, float, default, help, **kwargs)
    def add_str(switch, default, help, **kwargs):
        add_meta(switch, str, default, help, **kwargs)
    def test_bool(switch):
        add_bool(switch, False, '')
    add_int('--qcid',  None, 'query chip-id to investigate', nargs='*')
    add_int('--ocid',  [], 'query chip-id to investigate', nargs='*')
    add_int('--histid', None, 'history id (hard cases)', nargs='*')
    add_int('--r', [], 'view row', nargs='*')
    add_int('--c', [], 'view col', nargs='*')
    add_int('--nRows', 1, 'number of rows')
    add_int('--nCols', 1, 'number of cols')
    add_float('--xy-thresh', .001, '', step=.005)
    add_float('--ratio-thresh', 1.2, '', step=.1)
    add_int('--K', 10, 'for K-nearest-neighbors', step=20)
    add_str('--db', 'NAUTS', 'database to load')
    add_bool('--nopresent', default=False)
    add_bool('--save-figures', default=False)
    add_bool('--noannote', default=False)
    add_bool('--vrd', default=False)
    add_bool('--vcd', default=False)
    add_bool('--vrdq', default=False)
    add_bool('--vcdq', default=False)
    add_bool('--show-res', default=False)
    add_bool('--nocache-query', default=False)
    add_bool('--noprinthist', default=True)
    add_bool('--test-vsmany', default=False)
    add_bool('--test-vsone', default=False)

    add_str('--tests', [], 'integer or test name', nargs='*')

    add_str('--show-best', [], 'integer or test name', nargs='*')
    add_str('--show-worst', [], 'integer or test name', nargs='*')
    args, unknown = parser.parse_known_args()
    print('[invest] args    = %r' % (args,))
    print('[invest] unknown = %r' % (unknown,))
    print('[invest] ---------')
    print('==================')
    return args

args = parse_arguments()

import DataStructures as ds
import matching_functions as mf
import match_chips3 as mc3
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

def history_entry(database='', cid=-1, ocids=[], notes='', cx=-1):
    return (database, cid, ocids, notes)

# A list of poster child examples. (curious query cases)
GZ_greater1_cid_list = [140, 297, 306, 311, 425, 441, 443, 444, 445, 450, 451,
                        453, 454, 456, 460, 463, 465, 501, 534, 550, 662, 786,
                        802, 838, 941, 981, 1043, 1046, 1047]
HISTORY = [
    history_entry('GZ', 662,     [262],            notes='viewpoint / shadow (circle)'),
    history_entry('GZ', 1046,    [],               notes='extreme viewpoint #gt=2'),
    history_entry('GZ', 838,     [801, 980],       notes='viewpoint / quality'),
    history_entry('GZ', 501,     [140],            notes='dark lighting'),
    history_entry('GZ', 981,     [802],            notes='foal extreme viewpoint'),
    history_entry('GZ', 306,     [112],            notes='occlusion'),
    history_entry('GZ', 941,     [900],            notes='viewpoint / quality'),
    history_entry('GZ', 311,     [289],            notes='quality'),
    history_entry('GZ', 1047,    [],               notes='extreme viewpoint #gt=4'),
    history_entry('GZ', 297,     [301],            notes='quality'),
    history_entry('GZ', 786,     [787],            notes='foal #gt=11'),
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
    history_entry('NAUTS', 1,    [],               notes='simple eg'),
    history_entry('WDOGS', 1,    [],               notes='simple eg'),
    history_entry('MOTHERS', 69, [68],             notes='textured foal (lots of bad matches)'),
    history_entry('MOTHERS', 28, [27],             notes='viewpoint foal'),
    history_entry('MOTHERS', 53, [54],             notes='image quality'),
    history_entry('MOTHERS', 51, [50],             notes='dark lighting'),
    history_entry('MOTHERS', 44, [43, 45],         notes='viewpoint'),
    history_entry('MOTHERS', 66, [63, 62, 64, 65], notes='occluded foal'),
]

def mothers_problem_pairs():
    '''MOTHERS Dataset: difficult (qcx, cx) query/result pairs'''
    viewpoint = [( 16, 17), (19, 20), (73, 71), (75, 78), (108, 112), (110, 108)]
    quality = [(27, 26),  (52, 53), (67, 68), (73, 71), ]
    lighting = [(105, 104), ( 49,  50), ( 93,  94), ]
    confused = []
    occluded = [(64,65), ]
    return locals()

def quick_assign_vsmany(hs, qcx, cx, K): 
    print('[invest] Performing quick vsmany')
    cx2_desc = hs.feats.cx2_desc
    vsmany_args = hs.matcher.vsmany_args
    cx2_fm, cx2_fs, cx2_score = mc2.assign_matches_vsmany(qcx, cx2_desc, vsmany_args)
    fm = cx2_fm[cx]
    fs = cx2_fs[cx]
    return fm, fs

def quick_assign_vsone(hs, qcx, cx, **kwargs):
    print('[invest] Performing quick vsone')
    cx2_desc       = hs.feats.cx2_desc
    vsone_args     = mc2.VsOneArgs(cxs=[cx], **kwargs)
    vsone_assigned = mc2.assign_matches_vsone(qcx, cx2_desc, vsone_args)
    (cx2_fm, cx2_fs, cx2_score) = vsone_assigned
    fm = cx2_fm[cx] ; fs = cx2_fs[cx]
    return fm, fs

# Just put in PL

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

def investigate_scoring_rules(hs, qcx, fnum=1):
    args = hs.args
    K = 4 if args.K is None else args.K
    vr2.test_voting_rules(hs, qcx, K, fnum)
    fnum += 1
    return fnum

param1 = 'K'
param2 = 'xy_thresh'
assign_alg = 'vsmany'
nParam1=1 
fnum = 1
nParam2=1
cx_list='gt1'
        #fnum = vary_query_params(hs, qcx, 'ratio_thresh', 'xy_thresh', 'vsone', 4, 4, fnum, cx_list='gt1')
    #if '2' in args.tests:
        #hs.ensure_matcher(match_type='vsmany')
        #fnum = vary_query_params(hs, qcx, 'K', 'xy_thresh', 'vsmany', 4, 4, fnum, cx_list='gt1') 

def linear_logspace(start, stop, num, base=2):
    return 2 ** np.linspace(np.log2(start), np.log2(stop), num)


def vary_query_cfg(hs, qon_list, q_cfg=None, vary_cfg=None, fnum=1):
    if vary_cfg is None:
        vary_cfg = {'ratio_thresh' : [1.4, 1.6, 1.8], 
                       'xy_thresh' : [.001, .002, .01]}
    if q_cfg is None:
        q_cfg = mc3.get_vsone_cfg()
    # Ground truth matches
    for qcx, ocxs, notes in qon_list:
        gt_cxs = hs.get_groundtruth_cxs(qcx)
        for cx in gt_cxs:
            fnum = vary_two_cfg(hs, qcx, cx, notes, q_cfg, vary_cfg, fnum)
    return fnum

def vary_two_cfg(hs, qcx, cx, notes, q_cfg, vary_cfg, fnum=1):
    if len(vary_cfg) > 2:
        raise Exception('can only vary at most two cfgeters')
    print('[invest] vary_two_cfg: q'+hs.vs_str(qcx, cx))
    cfg_keys = vary_cfg.keys()
    cfg_vals = vary_cfg.values()
    cfg1_name = cfg_keys[0]
    cfg2_name = cfg_keys[1]
    cfg1_steps = cfg_vals[0]
    cfg2_steps = cfg_vals[1]
    nRows = len(cfg1_steps)
    nCols = len(cfg2_steps)

    print('[invest] Varying configs: nRows=%r, nCols=%r' % (nRows, nCols))
    print('[invest] %r = %r ' % (cfg1_name, cfg1_steps))
    print('[invest] %r = %r ' % (cfg2_name, cfg2_steps))
    ylabel_args = dict(rotation='horizontal',
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        fontproperties=df2.FONTS.medbold)
    xlabel_args = dict(fontproperties=df2.FONTS.medbold)
    #ax = df2.plt.gca()
    # Vary cfg1
    #df2.plt.gcf().clf()
    print_lock_ = helpers.ModulePrintLock(mc3, df2)
    assign_alg = q_cfg.a_cfg.query_type 
    for rowx, cfg1_value in enumerate(cfg1_steps):
        q_cfg.update_cfg(**{cfg1_name:cfg1_value})
        y_title = cfg1_name+'='+helpers.format(cfg1_value, 3)
        # Vary cfg2 
        for colx, cfg2_value in enumerate(cfg2_steps):
            q_cfg.update_cfg(**{cfg2_name:cfg2_value})
            plotnum = (nRows, nCols, rowx*nCols+colx+1)
            # HACK
            #print(plotnum)
            #print(q_cfg)
            # query only the chips of interest (groundtruth) when doing vsone
            if assign_alg == 'vsone':
                res = mc3.query_groundtruth(hs, qcx, q_cfg)
            # query the entire database in vsmany (just as fast as vgroundtruth)
            elif assign_alg == 'vsmany':
                res = mc3.query_database(hs, qcx, q_cfg)
            res.plot_matches(hs, cx, fnum=fnum, plotnum=plotnum,
                             show_gname=False, showTF=False)
            x_title = cfg2_name + '='+helpers.format(cfg2_value, 3)  #helpers.commas(cfg2_value, 3)
            ax = df2.plt.gca()
            if rowx == len(cfg1_steps) - 1:
                ax.set_xlabel(x_title, **xlabel_args)
            if colx == 0:
                ax.set_ylabel(y_title, **ylabel_args)
    del print_lock_
    vary_title = '%s vary %s and %s' % (assign_alg, cfg1_name, cfg2_name)
    figtitle =  '%s %s %s' % (vary_title, hs.vs_str(qcx, cx), notes)
    subtitle = mc3.simplify_test_uid(q_cfg.get_uid())
    df2.set_figtitle(figtitle, subtitle)
    df2.adjust_subplots_xylabels()
    fnum += 1
    viz.save_if_requested(hs, vary_title)
    return fnum

def quick_get_features_factory(hs):
    'builds a factory function'
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_cid  = hs.tables.cx2_cid 
    def get_features(cx):
        rchip = hs.get_chip(cx)
        fx2_kp = cx2_kpts[cx]
        fx2_desc = cx2_desc[cx]
        cid = cx2_cid[cx]
        return rchip, fx2_kp, fx2_desc, cid
    return get_features

def show_vsone_matches(hs, qcx, fnum=1):
    hs.ensure_matcher(match_type='vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    return res_vsone, fnum

def get_qfx2_gtkrank(hs, qcx, q_cfg): 
    '''Finds how deep possibily correct matches are in the ANN structures'''
    import matching_functions as mf
    dx2_cx   = q_cfg.data_index.ax2_cx
    gt_cxs   = hs.get_other_cxs(qcx)
    qcx2_nns = mf.nearest_neighbors(hs, [qcx], q_cfg)
    filt2_weights = mf.weight_neighbors(hs, qcx2_nns, q_cfg)
    K = q_cfg.nn_cfg.K
    (qfx2_dx, _) = qcx2_nns[qcx]
    qfx2_weights = filt2_weights['lnbnn'][qcx]
    qfx2_cx = dx2_cx[qfx2_dx[:, 0:K]]
    qfx2_gt = np.in1d(qfx2_cx.flatten(), gt_cxs)
    qfx2_gt.shape = qfx2_cx.shape
    qfx2_gtkrank = np.array([helpers.npfind(isgt) for isgt in qfx2_gt])
    qfx2_gtkweight = [0 if rank == -1 else weights[rank] 
                      for weights, rank in zip(qfx2_weights, qfx2_gtkrank)]
    qfx2_gtkweight = np.array(qfx2_gtkweight)
    return qfx2_gtkrank, qfx2_gtkweight

def measure_k_rankings(hs):
    'Reports the k match of correct feature maatches for each problem case'
    import match_chips3 as mc3
    K = 500
    q_cfg = mc3.QueryConfig(K=K, Krecip=0,
                                roidist_thresh=None, lnbnn_weight=1)
    id2_qcxs, id2_ocids, id2_notes = get_hard_cases(hs)
    id2_rankweight = [get_qfx2_gtkrank(hs, qcx, q_cfg) for qcx in id2_qcxs]
    df2.rrr()
    df2.reset()
    for qcx, rankweight, notes in zip(id2_qcxs, id2_rankweight, id2_notes):
        ranks, weights = rankweight
        ranks[ranks == -1] = K+1
        title = 'q'+hs.cxstr(qcx) + ' - ' + notes
        print(title)
        df2.figure(fignum=qcx, doclf=True, title=title)
        #draw_support
        #label=title
        df2.draw_hist(ranks, nbins=100, weights=weights) # FIXME
        df2.legend()
    print(len(id2_qcxs))
    df2.present(num_rc=(4,5), wh=(300,250))

def measure_cx_rankings(hs):
    ' Reports the best chip ranking over each problem case'
    import match_chips3 as mc3
    q_cfg = ds.QueryConfig(K=500, Krecip=0,
                                roidist_thresh=None, lnbnn_weight=1)
    id2_qcxs, id2_ocids, id2_notes = get_hard_cases(hs)
    id2_bestranks = []
    for id_ in xrange(len(id2_qcxs)):
        qcx = id2_qcxs[id_]
        reses = mc3.execute_query_safe(hs, q_cfg, [qcx])
        gt_cxs = hs.get_other_cxs(qcx)
        res = reses[2][qcx]
        cx2_score = res.get_cx2_score(hs)
        top_cxs  = cx2_score.argsort()[::-1]
        gt_ranks = [helpers.npfind(top_cxs == gtcx) for gtcx in gt_cxs]
        bestrank = min(gt_ranks)
        id2_bestranks += [bestrank]
    print(id2_bestranks)

def plot_name(hs, qcx, fnum=1, **kwargs):
    print('[invest] Plotting name')
    viz.plot_name_of_cx(hs, qcx, fignum=fnum, **kwargs)
    return fnum+1

def show_names(hs, qon_list, fnum=1):
    '''The most recent plot names function, works with qon_list'''
    args = hs.args
    result_dir = hs.dirs.result_dir
    names_dir = join(result_dir, 'show_names')
    helpers.ensuredir(names_dir)
    for (qcx, ocxs, notes) in qon_list:
        print('Showing q%s - %r' % (hs.cxstr(qcx), notes))
        fnum = plot_name(hs, qcx, fnum, subtitle=notes, annote=not args.noannote)
        if args.save_figures:
            df2.save_figure(fpath=names_dir, usetitle=True)
    return fnum

def compare_matching_methods(hs, qcx, fnum=1):
    print('[invest] Comparing match methods')
    # VSMANY matcher
    hs.ensure_matcher(match_type='vsmany')
    vsmany_score_options = ['LNRAT', 'LNBNN', 'RATIO']
    vsmany_args = hs.matcher.vsmany_args
    vsmany_results = {}
    for score_type in vsmany_score_options:
        params.__VSMANY_SCORE_FN__ = score_type
        res_vsmany = mc2.build_result_qcx(hs, qcx)
        df2.show_match_analysis(hs, res_vsmany, N=5, fignum=fnum, figtitle=' LNRAT')
        vsmany_results[score_type] = res_vsmany
        fnum+=1
    # BAGOFWORDS matcher
    hs.ensure_matcher(match_type='bagofwords')
    resBOW = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resBOW, N=5, fignum=fnum, figtitle=' bagofwords')
    fnum+=1
    # VSONE matcher
    hs.ensure_matcher(match_type='vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    # Extra 
    df2.show_match_analysis(hs, vsmany_results['LNBNN'], N=20, fignum=fnum,
                                figtitle=' LNBNN More', show_query=False)
    fnum+=1
    return fnum

# ^^^^^^^^^^^^^^^^^
# Tests
    
def hs_from_db(db, args=None):
    # Load hotspotter
    db_dir = eval('params.'+db)
    print('[invest] loading hotspotter database')
    def _hotspotter_load():
        hs = ld2.HotSpotter()
        hs.load_all(db_dir, matcher=False, args=args)
        hs.set_samples()
        return hs
    if True:
        hs = _hotspotter_load()
        return hs
    else:
        rss = helpers.RedirectStdout(autostart=True)
        hs = _hotspotter_load()
        rss.stop()
        rss.dump()
        print('loaded hotspotter database')
        return hs

def get_hard_cases(hs):
    qcid_list = []
    ocid_list = []
    note_list = []
    db = hs.args.db
    for (db_, qcid, ocids, notes) in HISTORY:
        if db == db_:
            qcid_list += [qcid]
            ocid_list += [ocids]
            note_list += [notes]
    qcx_list = hs.cid2_cx(qcid_list)
    #print('qcid_list = %r ' % qcid_list)
    #print('qcx_list = %r ' % qcid_list)
    #print('[get_hard_cases]\n %r\n %r\n %r\n' % (qcx_list, ocid_list, note_list))
    return qcx_list, ocid_list, note_list

def view_all_history_names_in_db(hs, db):
    qcx_list = zip(*get_hard_cases(hs))[0]
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

def vary_vsone_cfg(hs, qon_list, fnum, vary_dicts, **kwargs):
    vary_cfg = helpers.dict_union(*vary_dicts)
    q_cfg = mc3.get_vsone_cfg(**kwargs)
    return vary_query_cfg(hs, qon_list, q_cfg, vary_cfg, fnum)

def vary_vsmany_cfg(hs, qon_list, vary_cfg, fnum, **kwargs):
    vary_cfg = helpers.dict_union(*vary_dicts)
    q_cfg = mc3.get_vsmany_cfg(**kwargs)
    return vary_query_cfg(hs, qon_list, q_cfg, vary_cfg, fnum)

def run_investigations(hs, qon_list):
    args = hs.args
    qcx = qon_list[0][0]
    print('[invest] Running Investigation: '+hs.cxstr(qcx))
    fnum = 1
    #view_all_history_names_in_db(hs, 'MOTHERS')
    #fnum = compare_matching_methods(hs, qcx, fnum)
    #xy_  = {'xy_thresh'    : [None, .2, .02, .002]}
    xy_  = {'xy_thresh'    : [None, .02, .002]}
    #rat_ = {'ratio_thresh' : [None, 1.4, 1.6, 1.8]}
    rat_ = {'ratio_thresh' : [None, 1.5, 1.7]}
    K_   = {'K'            : [2, 5, 10]}
    Kr_  = {'Krecip'       : [0, 2, 5, 10]}
    if '0' in args.tests or 'show-names' in args.tests:
        show_names(hs, qon_list)
    if '1' in args.tests or 'vary-vsone-rat-xy' in args.tests:
        fnum = vary_vsone_cfg(hs, qon_list, fnum, [rat_, xy_])
    if '2' in args.tests or 'vary-vsmany-k-xy' in args.tests:
        fnum = vary_vsmany_cfg(hs, qon_list, fnum, [K_, xy_])
    if '3' in args.tests:
        fnum = vary_query_cfg(hs, qon_list, fnum, [K_, Kr_], sv_on=True) 
        fnum = vary_query_cfg(hs, qon_list, fnum, [K_, Kr_], sv_on=False) 
    if '4' in args.tests:
        fnum = investigate_scoring_rules(hs, qcx, fnum)
    if '6' in args.tests:
        measure_k_rankings(hs)
    if '7' in args.tests:
        measure_cx_rankings(hs)
    if '8' in args.tests:
        mc3.compare_scoring(hs)
    if '9' in args.tests:
        fnum = plot_keypoint_scales(hs)
    if '10' in args.tests or 'vsone-gt' in args.tests:
        fnum = investigate_vsone_groundtruth(hs, qon_list, fnum)
    if '11' in args.tests:
        fnum = investigate_chip_info(hs, qon_list, fnum)
    if '12' in args.tests or 'test-cfg-vsone-1' in args.tests:
        import dev
        dev.test_configurations(hs, qon_list, ['vsone_1'])

#===========
# Main Script
# exec(open('investigate_chip.py').read())

def print_history_table():
    print('------------')
    print('[invest] Printing history table:')
    count = 0
    for histentry in HISTORY:
        if args.db == histentry[0]:
            print('%d: %r' % (count, histentry))
            count += 1

def change_db(db):
    global args
    global hs
    args.db = db
    hs = hs_from_db(args.db, args)
    hs.args = args

def main():
    print('[iv] main()')
    if 'hs' in vars():
        return
    # Load Hotspotter
    hs = hs_from_db(args.db, args)
    qon_list = get_qon_list(hs)
    print('[invest] Loading DB=%r' % args.db)
    if not args.noprinthist or True:
        print('---')
        print('[invest] print_history_table()')
        print_history_table()
    qcxs_list, ocxs_list, notes_list = zip(*qon_list)
    qcxs  = qcxs_list[0]
    notes = notes_list[0]
    print('========================')
    print('[invest] Loaded DB=%r' % args.db)
    print('[invest] notes=%r' % notes)
    qcxs = helpers.ensure_iterable(qcxs)
    return locals()

def plot_keypoint_scales(hs, fnum=1):
    print('[invest] plot_keypoint_scales()')
    cx2_kpts = hs.feats.cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[invest] num_keypoints = %r ' % len(kpts))
    print('[invest] keypoints per image stats = '+helpers.printable_mystats(cx2_nFeats)) 
    acd = kpts[:,2:5].T
    det = 1/(acd[0] * acd[2])
    sdet = np.array(sorted(det))
    print('scale stats: '+helpers.printable_mystats(sdet))
    #
    fig = df2.figure(fignum=fnum, doclf=True)
    df2.plot(sdet)
    ax = df2.plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    #
    fnum += 1
    fig = df2.figure(fignum=fnum, doclf=True)
    df2.show_histogram(sdet, bins=20)
    ax = df2.plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fnum

def get_qon_list(hs):
    print('[invest] get_qon_list()')
    # Get query ids
    qon_list = []
    histids = None if args.histid is None else np.array(args.histid)
    if args.qcid is None:
        qon_hard = zip(*get_hard_cases(hs))
        if histids is None:
            print('[invest] Chosen all hard histids')
            qon_list += qon_hard
        elif not histids is None:
            print('[invest] Chosen histids=%r' % histids)
            qon_list += [qon_hard[id_] for id_ in histids]
    else:
        print('[invest] Chosen qcid=%r' % args.qcid)
        qcx_list =  helpers.ensure_iterable(hs.cid2_cx(args.qcid))
        ocid_list = [[]*len(qcx_list)]
        note_list = [['user selected qcid']*len(qcx_list)]
        qon_list += [zip(qcx_list, ocid_list, note_list)]
    return qon_list

def investigate_vsone_groundtruth(hs, qon_list, fnum=1):
    print('--------------------------------------')
    print('[invest] investigate_vsone_groundtruth')
    q_cfg = mc3.get_vsone_cfg(sv_on=True, ratio_thresh=1.5)
    for qcx, ocxs, notes in qon_list:
        res = mc3.query_groundtruth(hs, qcx, q_cfg)
        #print(q_cfg)
        #print(res)
        #res.show_query(hs, fignum=fnum)
        fnum += 1
        res.show_topN(hs, fignum=fnum, q_cfg=q_cfg)
        fnum += 1
    return fnum

def chip_info(hs, cx, notes=''):
    nx = hs.tables.cx2_nx[cx]
    gx = hs.tables.cx2_gx[cx]
    name = hs.tables.nx2_name[nx]
    gname = hs.tables.gx2_gname[gx]
    indexed_gt_cxs = hs.get_other_indexed_cxs(cx)
    gt_cxs = hs.get_other_cxs(cx)
    print('[invest] Chip Info ')
    infostr_list = [
        hs.cxstr(cx),
        'notes=%r' % notes,
        'cx=%r' % cx,
        'gx=%r' % gx,
        'nx=%r' % nx,
        'name=%r' % name,
        'gname=%r' % gname,
        'nGroundTruth = %s ' % str(len(gt_cxs)),
        'nIndexedGroundTruth = %s ' % str(len(indexed_gt_cxs)),
        'Ground Truth: %s' % (hs.cx_liststr(gt_cxs),),
        'IndexedGroundTruth = %s' % (hs.cx_liststr(indexed_gt_cxs),),
    ]
    print(helpers.indent('\n'.join(infostr_list), '[invest] '))

def investigate_chip_info(hs, qon_list, fnum=1):
    for qcx, ocxs, notes in qon_list:
        chip_info(hs, qcx, notes)
    return fnum

if __name__ == '__main__':
    print('[invest] __main__ ')
    df2.DARKEN = .5
    main_locals = main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    fmtstr = helpers.make_progress_fmt_str(len(qcxs), '[invest] investigation ')
    print('[invest]====================')
    for count, qcx in enumerate(qcxs):
        print(fmtstr % (count+1))
        run_investigations(hs, qon_list)
    print('[invest]====================')
    kwargs = {}
    dcxs = None
    q_cfg = None
    #df2.update()
    if hs.args.nopresent:
        print('...not presenting')
        sys.exit(0)
    exec(df2.present()) #**df2.OooScreen2()
