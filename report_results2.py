import drawing_functions2 as df2
import helpers
import numpy as np
import datetime
import textwrap
import sys
from os.path import realpath, join

def printDBG(msg):
    #print(msg)
    pass

# ========================================================
# Driver functions (reports results for entire experiment)
# ========================================================

def write_rank_results(hs, qcx2_res):
    rankres_str = rank_results(hs, qcx2_res)
    report_type = 'rank_'
    write_report(hs, rankres_str, report_type)

def write_report(hs, report_str, report_type):
    result_dir = hs.dirs.result_dir
    algo_uid   = hs.algo_uid()
    timestamp  = get_timestamp()
    csv_fname  = algo_uid+report_type+timestamp+'.csv'
    helpers.ensurepath(result_dir)
    rankres_csv = join(result_dir, csv_fname)
    helpers.write_to(rankres_csv, report_str)
    if '--gvim' in sys.argv:
        helpers.gvim(rankres_csv)

def get_true_positive_ranks(qcx, top_cx, cx2_nx):
    'Returns the ranking of the other chips which should have scored high'
    top_nx = cx2_nx[top_cx]
    qnx    = cx2_nx[qcx]
    _truepos_ranks, = np.where(top_nx == qnx)
    truepos_ranks = _truepos_ranks[top_cx[_truepos_ranks] != qcx]
    return truepos_ranks

def get_false_positive_ranks(qcx, top_cx, cx2_nx):
    'Returns the ranking of the other chips which should have scored high'
    top_nx = cx2_nx[top_cx]
    qnx    = cx2_nx[qcx]
    _falsepos_ranks, = np.where(top_nx != qnx)
    falsepos_ranks = _falsepos_ranks[top_cx[_falsepos_ranks] != qcx]
    return falsepos_ranks

def rank_results(hs, qcx2_res):
    cx2_cid  = hs.tables.cx2_cid
    cx2_nx   = hs.tables.cx2_nx
    nx2_name = hs.tables.nx2_name
    cx2_top_truepos_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_truepos_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_score         = np.zeros(len(cx2_cid)) - 100

    for qcx, qcid in enumerate(cx2_cid):
        res = qcx2_res[qcx]
        if res.cx2_fs is None or len(res.cx2_fs) == 0: continue
        # The score is the sum of the feature scores
        cx2_score = np.array([np.sum(fs) for fs in res.cx2_fs])
        top_cx = np.argsort(cx2_score)[::-1]
        top_score = cx2_score[top_cx]
        # Get true postiive ranks
        truepos_ranks = get_true_positive_ranks(qcx, top_cx, cx2_nx)
        # Find statitics about the true positives (and negatives)
        if len(truepos_ranks) > 0:
            top_truepos_rank = truepos_ranks.min()
            bot_truepos_rank = truepos_ranks.max()
            true_neg_range   = np.arange(0, bot_truepos_rank+2)
            top_trueneg_rank = np.setdiff1d(true_neg_range, truepos_ranks).min()
            top_trupos_score = top_score[top_truepos_rank]
        else:
            top_trueneg_rank = 0
            top_truepos_rank = np.NAN
            top_trupos_score = np.NAN
        # Append stats to output
        cx2_top_truepos_rank[qcx]  = top_truepos_rank
        cx2_top_truepos_score[qcx] = top_trupos_score
        cx2_top_trueneg_rank[qcx]  = top_trueneg_rank
        cx2_top_trueneg_score[qcx] = top_score[top_trueneg_rank]
        cx2_top_score[qcx]         = top_score[0]
    # difference between the top score and the actual best score
    cx2_score_disp = cx2_top_score - cx2_top_truepos_score
    #
    # Easy to digest results
    num_chips = len(cx2_top_truepos_rank)
    num_with_gtruth = (1 - np.isnan(cx2_top_truepos_rank)).sum()
    num_rank_less5 = (cx2_top_truepos_rank < 5).sum()
    num_rank_less1 = (cx2_top_truepos_rank < 1).sum()
    
    # Output ranking results

    # Build the experiment csv metadata
    rankres_metadata = textwrap.dedent('''
    # Rank Result Metadata:
    #   CID        = Query chip-id
    #   TT RANK   = top true positive rank
    #   TF RANK   = top false positive rank
    #   TT SCORE  = top true positive score
    #   SCORE DISP = disparity between top-score and top-true-positive-score
    #   NAME       = Query chip-name''').strip()

    # Build the experiemnt csv header
    rankres_csv_header = '#CID,  TT RANK,  TF RANK,  TT SCORE,  TF SCORE, SCORE DISP, NX'

    # Build the experiment csv data lines
    todisp = np.vstack([cx2_cid,
                        cx2_top_truepos_rank,
                        cx2_top_trueneg_rank,
                        cx2_top_truepos_score,
                        cx2_top_trueneg_score,
                        cx2_score_disp, 
                        cx2_nx]).T
    rankres_csv_lines = []
    for (cid, ttr, ttnr, tts, ttns, sdisp, nx) in todisp:
        csv_line = ('%4d, %8.0f, %8.0f, %9.2f, %9.2f, %10.2f, %4d' %\
              (cid, ttr, ttnr, tts, ttns, sdisp, nx) )
        rankres_csv_lines.append(csv_line)

    # Build the experiment summary report
    rankres_summary  = '\n'
    rankres_summary += '# Experiment Settings (hs.algo_uid): '+hs.algo_uid()+'\n'
    rankres_summary +=  get_timestamp(format='comment')+'\n'
    rankres_summary += '# Num Chips: %d \n' % num_chips
    rankres_summary += '# Num Chips with at least one match: %d \n' % num_with_gtruth
    rankres_summary += '# Ranks <= 5: %d / %d\n' % (num_rank_less5, num_with_gtruth)
    rankres_summary += '# Ranks <= 1: %d / %d' % (num_rank_less1, num_with_gtruth)

    print(rankres_summary)

    # Concateate parts into a csv file and return

    rankres_csv_str = '\n'.join(rankres_csv_lines)
    rankres_str = '\n'.join([rankres_summary+'\n', 
                             rankres_metadata+'\n',
                             rankres_csv_header,
                             rankres_csv_str])

    return rankres_str

def get_timestamp(format='filename'):
    now = datetime.datetime.now()
    time_tup = (now.year, now.month, now.day, now.hour, now.minute)
    time_formats = {
        'filename': 'ymd-%04d-%02d-%02d_hm-%02d-%02d',
        'comment' : '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d' }
    stamp = time_formats[format] % time_tup
    return stamp

def cx2_other_cx(hs, cx):
    cx2_nx   = hs.tables.cx2_nx
    nx = cx2_nx[cx]
    other_cx_, = np.where(cx2_nx == nx)
    other_cx  = other_cx_[other_cx_ != cx]
    return other_cx

def get_oxsty_mAP_score(hs, res, SV):
    qcx = res.qcx
    qnx = hs.tables.cx2_nx[qcx]
    cx2_score, cx2_fm, cx2_fs = res.get_info(SV)
    top_cx = cx2_score.argsort()[::-1]
    top_gx = hs.tables.cx2_gx[top_cx]
    top_gname = hs.tables.gx2_gname[top_gx]


    ground_truth_query = 
    ranked_list = [top_gname.replace('.jpg')]
    

def print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=False):
    res = qcx2_res[qcx]
    print_top_res_scores(hs, res, view_top, SV)

def print_top_res_scores(hs, res, view_top=10, SV=False):
    qcx = res.qcx
    cx2_score, cx2_fm, cx2_fs = res.get_info(SV)
    lbl = ['(assigned)', '(assigned+V)'][SV]
    cx2_nx     = hs.tables.cx2_nx
    nx2_name   = hs.tables.nx2_name
    qnx        = cx2_nx[qcx]
    other_cx   = cx2_other_cx(hs, qcx)
    top_cx     = cx2_score.argsort()[::-1]
    top_scores = cx2_score[top_cx] 
    top_nx     = cx2_nx[top_cx]
    view_top   = min(len(top_scores), view_top)
    print('---------------------------------------')
    print('Inspecting matches of qcx=%d name=%s' % (qcx, nx2_name[qnx]))
    print(' * Matched against %d other chips' % len(cx2_score))
    print(' * Ground truth chip indexes:\n   other_cx=%r' % other_cx)
    print('The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('--> cx=%4d, score=%6.2f' % (cx, score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print(_mark+' cx=%4d, score=%6.2f' % (tcx, tscore))
    print('---------------------------------------')
    print('---------------------------------------')

def get_tp_matches(res, hs, SV):
    qcx = res.qcx
    cx2_nx = hs.tables.cx2_nx
    cx2_score, cx2_fm, cx2_fs = res.get_info(SV)
    top_cx = np.argsort(cx2_score)[::-1]
    top_score = cx2_score[top_cx]
    # Get true postive ranks (groundtruth)
    truepos_ranks  = get_true_positive_ranks(qcx, top_cx, cx2_nx)
    truepos_scores = top_score[truepos_ranks]
    truepos_cxs    = top_cx[truepos_ranks]
    return truepos_cxs, truepos_ranks, truepos_scores

def get_fp_matches(res, hs, SV):
    qcx = res.qcx
    cx2_nx = hs.tables.cx2_nx
    cx2_score, cx2_fm, cx2_fs = res.get_info(SV)
    top_cx = np.argsort(cx2_score)[::-1]
    top_score = cx2_score[top_cx]
    # Get false postive ranks (non-groundtruth)
    falsepos_ranks  = get_false_positive_ranks(qcx, top_cx, cx2_nx)
    falsepos_scores = top_score[falsepos_ranks]
    falsepos_cxs    = top_cx[falsepos_ranks]
    return falsepos_cxs, falsepos_ranks, falsepos_scores

def get_nth_truepos_match(res, hs, n, SV):
    truepos_cxs, truepos_ranks, truepos_scores = get_tp_matches(res, hs, SV)
    nth_cx    = truepos_cxs[n]
    nth_rank  = truepos_ranks[n]
    nth_score = truepos_scores[n]
    printDBG('Getting the nth=%r true pos cx,rank,score=(%r, %r, %r)' % \
          (n, nth_cx, nth_rank, nth_score))
    return nth_cx, nth_rank, nth_score

def get_nth_falsepos_match(res, hs, n, SV):
    falsepos_cxs, falsepos_ranks, falsepos_scores = get_fp_matches(res, hs, SV)
    nth_cx    = falsepos_cxs[n]
    nth_rank  = falsepos_ranks[n]
    nth_score = falsepos_scores[n]
    printDBG('Getting the nth=%r false pos cx,rank,score=(%r, %r, %r)' % \
          (n, nth_cx, nth_rank, nth_score))
    return nth_cx, nth_rank, nth_score


def get_tt_bt_tf_cxs(hs, res, SV):
    'Returns the top and bottom true positives and top false positive'
    qcx = res.qcx
    tt_cx, tt_rank, tt_score = get_nth_truepos_match(res,  hs,  0, SV)
    bt_cx, bt_rank, bt_score = get_nth_truepos_match(res,  hs, -1, SV)
    tf_cx, tf_rank, tf_score = get_nth_falsepos_match(res, hs,  0, SV)
    titles = ('TopTP rank='+str(tt_rank)+' ',
              'BotTP rank='+str(bt_rank)+' ',
              'TopFP rank='+str(tf_rank)+' ')
    cxs = (tt_cx, bt_cx, tf_cx)
    return cxs, titles

def visualize_res_tt_bt_tf(hs, res):
    #print('Visualizing result: ')
    #res.printme()
    SV = False
    qcx = res.qcx
    _fn = qcx
    cxs, titles = get_tt_bt_tf_cxs(hs, res, SV)
    df2.show_matches3(res, hs, cxs[0], SV, fignum=_fn+.231, title_aug=titles[0])
    df2.show_matches3(res, hs, cxs[1], SV, fignum=_fn+.232, title_aug=titles[1])
    df2.show_matches3(res, hs, cxs[2], SV, fignum=_fn+.233, title_aug=titles[2])
    SV = True
    cxsV, titlesV = get_tt_bt_tf_cxs(hs, res, SV)
    df2.show_matches3(res, hs, cxsV[0], SV, fignum=_fn+.234, title_aug=titlesV[0])
    df2.show_matches3(res, hs, cxsV[1], SV, fignum=_fn+.235, title_aug=titlesV[1])
    df2.show_matches3(res, hs, cxsV[2], SV, fignum=_fn+.236, title_aug=titlesV[2])
    df2.set_figtitle('fig '+str(_fn)+' -- ' + hs.query_uid())

def visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx):
    res = qcx2_res[qcx]
    visualize_res_tt_bt_tf(hs, res)


def visualize_all_qcx_tt_bt_tf(hs, qcx2_res):
    for qcx, res in enumerate(qcx2_res):
        visualize_res_tt_bt_tf(res)


def compute_average_precision(res, k):
    '''
    % from wikipedia: http://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
    $p(r)$ = precision as a function of recall $r$
    $AveP$ = $\int_0^1 p(r) dr$ = $\sum_{k=1}^n P(k) \del r(k)$

    $k$ is the rank in sequence of retrieved documents
    $n$ is the number of retrieved documents

    $\del r(k) = r(k) - r(k-1)$ = change in recall 
    '''
    pass

def compute_mean_average_precision(res, k):
    '''
    MAP = 1/Q \sum_{q=1}^Q AveP(q) 
    '''

# Score a single query for name consistency
# Written: 5-28-2013 
def res2_name_consistency(hs, res):
    '''Score a single query for name consistency
    Input: 
        res - query result
    Returns: Dict
        error_chip - degree of chip error
        name_error - degree of name error
        gt_pos_name - 
        gt_pos_chip - 
    '''
    # Defaults to -1 if no ground truth is in the top results
    cm, nm = em.hs.get_managers('cm','nm')
    qcx  = res.rr.qcx
    qnid = res.rr.qnid
    qnx   = nm.nid2_nx[qnid]
    ret = {'name_error':-1,      'chip_error':-1,
           'gt_pos_chip':-1,     'gt_pos_name':-1, 
           'chip_precision': -1, 'chip_recall':-1}
    if qnid == nm.UNIDEN_NID: exec('return ret')
    # ----
    # Score Top Chips
    top_cx = res.cx_sort()
    gt_pos_chip_list = (1+pylab.find(qnid == cm.cx2_nid(top_cx)))
    # If a correct chip was in the top results
    # Reward more chips for being in the top X
    if len(gt_pos_chip_list) > 0:
        # Use summation formula sum_i^n i = n(n+1)/2
        ret['gt_pos_chip'] = gt_pos_chip_list.min()
        _N = len(gt_pos_chip_list)
        _SUM_DENOM = float(_N * (_N + 1)) / 2.0
        ret['chip_error'] = float(gt_pos_chip_list.sum())/_SUM_DENOM
    # Calculate Precision / Recall (depends on the # threshold/max_results)
    ground_truth_cxs = np.setdiff1d(np.array(nm.nx2_cx_list[qnx]), np.array([qcx]))
    true_positives  = top_cx[gt_pos_chip_list-1]
    false_positives = np.setdiff1d(top_cx, true_positives)
    false_negatives = np.setdiff1d(ground_truth_cxs, top_cx)

    nTP = float(len(true_positives)) # Correct result
    nFP = float(len(false_positives)) # Unexpected result
    nFN = float(len(false_negatives)) # Missing result
    #nTN = float( # Correct absence of result

    ret['chip_precision'] = nTP / (nTP + nFP)
    ret['chip_recall']    = nTP / (nTP + nFN)
    #ret['true_negative_rate'] = nTN / (nTN + nFP)
    #ret['accuracy'] = (nTP + nFP) / (nTP + nTN + nFP + nFN)
    # ----
    # Score Top Names
    (top_nx, _) = res.nxcx_sort()
    gt_pos_name_list = (1+pylab.find(qnid == nm.nx2_nid[top_nx]))
    # If a correct name was in the top results
    if len(gt_pos_name_list) > 0: 
        ret['gt_pos_name'] = gt_pos_name_list.min() 
        # N should always be 1
        _N = len(gt_pos_name_list)
        _SUM_DENOM = float(_N * (_N + 1)) / 2.0
        ret['name_error'] = float(gt_pos_name_list.sum())/_SUM_DENOM
    # ---- 
    # RETURN RESULTS
    return ret

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    import match_chips2 as mc2
    import load_data2
    import imp
    #imp.reload(df2)
    #imp.reload(mc2)
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.DEFAULT
    hs = mc2.HotSpotter(db_dir)
    df2.close_all_figures()
    try:
        qcx2_res = mc2.run_matching(hs)
        qcx = 1
        #print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=False)
        #print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=True)
        visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx)

        def dinspect(qcx):
            df2.close_all_figures()
            visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx)
            df2.present()

    except Exception as ex:
        print(repr(ex))
        raise

    'dev inspect'

    # Execing df2.present does an IPython aware plt.show()
    exec(df2.present(wh=(900,600)))
