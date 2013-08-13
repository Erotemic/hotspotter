import drawing_functions2 as df2
import load_data2
import subprocess
import params
import helpers
import numpy as np
import datetime
import textwrap
import os
import sys
from os.path import realpath, join

def printDBG(msg):
    #print(msg)
    pass

# ========================================================
# Driver functions (reports results for entire experiment)
# ========================================================

def write_rank_results(hs, qcx2_res, SV=True):
    rankres_str = rank_results(hs, qcx2_res, SV)
    write_report(hs, rankres_str, 'rank_', SV)

def write_oxsty_mAP_results(hs, qcx2_res, SV=True):
    oxsty_map_csv = oxsty_mAP_results(hs, qcx2_res, SV)
    write_report(hs, oxsty_map_csv, 'oxsty-mAP_', SV)

def dump_qcx_tt_bt_tf(hs, qcx2_res):
    print(textwrap.dedent('''
    =============================
    Dumping results
    ============================='''))
    dump_dir = join(hs.dirs.result_dir, 'tt_bt_tf')
    helpers.ensurepath(dump_dir)
    if '--vd' in sys.argv:
        helpers.vd(dump_dir)
    for qcx in hs.test_sample_cx:
        df2.close_all_figures()
        res = qcx2_res[qcx]
        visualize_res_tt_bt_tf(hs, res)
        fig_fname = 'ttbttf_qcx' + str(qcx) + '--' + params.get_query_uid() + '.jpg'
        fig_fpath = join(dump_dir, fig_fname)
        print fig_fpath
        df2.save_figure(qcx, fig_fpath)
    df2.close_all_figures()
    return dump_dir

def write_report(hs, report_str, report_type, SV):
    result_dir = hs.dirs.result_dir
    query_uid  = params.get_query_uid()
    timestamp  = get_timestamp()
    SV_aug = ['_SVoff_','_SVon_'][SV] #TODO: SV should go into params
    csv_fname  = report_type+query_uid+SV_aug+timestamp+'.csv'
    helpers.ensurepath(result_dir)
    rankres_csv = join(result_dir, csv_fname)
    helpers.write_to(rankres_csv, report_str)
    if '--gvim' in sys.argv:
        helpers.gvim(rankres_csv)


# TODO: Better plots
# TODO: Make SV a parameter?
'''
Stem Plot of ranks
Output the false matches which score higher than the true matches
the number of ground truth images. The ranks of each. 
The hard pairs

For each Chip: 
    All Good Matches
    All Bad  Matches

Lowest Scoring Correct Matches
Highest Scoring Incorrect Matches
Disparity between the two. 
'''
def plot_stem(hs, qcx2_res, SV):
    pass


def rank_results(hs, qcx2_res, SV):
    'Builds csv files showing the cxs/scores/ranks of the query results'
    cx2_cid  = hs.tables.cx2_cid
    #---
    qcx2_top_true_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_top_true_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_top_true_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    qcx2_bot_true_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_bot_true_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_bot_true_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    qcx2_top_false_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_top_false_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_top_false_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    test_sample_cx = hs.test_sample_cx
    db_sample_cx   = hs.database_sample_cx
    for qcx in iter(test_sample_cx):
        res = qcx2_res[qcx]
        cx2_score = res.cx2_score_V if SV else res.cx2_score
        # The score is the sum of the feature scores
        top_cx = np.argsort(cx2_score)[::-1]
        top_score = cx2_score[top_cx]
        # Get true postiive ranks
        true_tup, false_tup = get_matchs_true_and_false(hs, res, SV)
        (true_cxs,  true_scores,  true_ranks)  = true_tup
        (false_cxs, false_scores, false_ranks) = false_tup
        nth_true  = lambda n: (true_cxs[n],  true_ranks[n],  true_scores[n])
        nth_false = lambda n: (false_cxs[n], false_ranks[n], false_scores[n])
        # Find statitics about the true positives (and negatives)
        if len(true_ranks) == 0:
            tt_cx, tt_r, tt_s = (np.nan, np.nan, np.nan)
            bt_cx, bt_r, bt_s = (np.nan, np.nan, np.nan)
        else:
            tt_cx, tt_r, tt_s = nth_true(0)
            bt_cx, bt_r, bt_s = nth_true(-1)
        tf_cx, tf_r, tf_s = nth_false(0)
        # Append stats to output
        qcx2_top_true_rank[qcx]   = tt_r
        qcx2_top_true_score[qcx]  = tt_s
        qcx2_top_true_cx[qcx]     = tt_cx
        #
        qcx2_bot_true_rank[qcx]   = bt_r
        qcx2_bot_true_score[qcx]  = bt_s
        qcx2_bot_true_cx[qcx]     = bt_cx
        #
        qcx2_top_false_rank[qcx]  = tf_r
        qcx2_top_false_score[qcx] = tf_s
        qcx2_top_false_cx[qcx]    = tf_cx
    # Easy to digest results
    num_chips = len(test_sample_cx)
    num_nonquery = len(np.setdiff1d(db_sample_cx, test_sample_cx))
    num_with_gtruth = (1 - np.isnan(qcx2_top_true_rank[test_sample_cx])).sum()
    num_rank_less5 = (qcx2_top_true_rank[test_sample_cx] < 5).sum()
    num_rank_less1 = (qcx2_top_true_rank[test_sample_cx] < 1).sum()
    # Output ranking results
    # TODO: mAP score
    # Build the experiment csv metadata
    SV_aug = ['_SVoff_','_SVon_'][SV] #TODO: SV should go into params
    query_uid = params.get_query_uid()+SV_aug
    header = '# Experiment Settings (params.query_uid):'+query_uid+'\n'
    header +=  get_timestamp(format='comment')+'\n'
    header += '# Num Query Chips: %d \n' % num_chips
    header += '# Num Query Chips with at least one match: %d \n' % num_with_gtruth
    header += '# Num NonQuery Chips: %d \n' % num_nonquery
    header += '# Ranks <= 5: %d / %d\n' % (num_rank_less5, num_with_gtruth)
    header += '# Ranks <= 1: %d / %d\n\n' % (num_rank_less1, num_with_gtruth)
    #---
    header += '# Full Parameters: \n' + helpers.indent(params.param_string(),'#') + '\n\n'
    #---
    header += textwrap.dedent('''
    # Rank Result Metadata:
    #   QCX  = Query chip-index
    #   TT   = top true  
    #   BT   = bottom true
    #   TF   = top false''').strip()
    # Build the experiemnt csv header
    column_labels = ['QCX', 'TT RANK', 'TT SCORE', 'TT CX', 'BT RANK', 
                     'BT SCORE', 'BT CX', 'TF RANK', 'TF SCORE', 'TF CX']
    column_list = [test_sample_cx, 
                   qcx2_top_true_rank[test_sample_cx],
                   qcx2_top_true_score[test_sample_cx],
                   qcx2_top_true_cx[test_sample_cx],
                   qcx2_bot_true_rank[test_sample_cx],
                   qcx2_bot_true_score[test_sample_cx],
                   qcx2_bot_true_cx[test_sample_cx],
                   qcx2_top_false_rank[test_sample_cx],
                   qcx2_top_false_score[test_sample_cx],
                   qcx2_top_false_cx[test_sample_cx]]
    column_type = [int, int, float, int, int, float, int, int, float, int]
    rankres_str = load_data2.make_csv_table(column_labels, column_list, header, column_type)
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

def dump_top_images(hs, qcx2_res):
    for qcx in hs.test_sample_cx:
        res = qcx2_res[qcx]
    df2.show_matches3(res, hs, cxs[0], SV, fignum=_fn+.231, title_aug=titles[0])

def print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=False):
    res = qcx2_res[qcx]
    print_top_res_scores(hs, res, view_top, SV)

def print_top_res_scores(hs, res, view_top=10, SV=False):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
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

# OXFORD STUFF
def oxsty_mAP_results(hs, qcx2_res, SV):
    # Check directorys where ranked lists of images names will be put
    SV_aug = ['_SVoff_','_SVon_'][SV] #TODO: SV should go into params
    qres_dir  = hs.dirs.qres_dir
    query_uid = params.get_query_uid()
    oxsty_qres_dname = 'oxsty_qres' + query_uid + SV_aug
    oxsty_qres_dpath = join(qres_dir, oxsty_qres_dname)
    helpers.ensure_path(oxsty_qres_dpath)
    # Get the mAP scores using philbins program
    query_mAP_list = []
    query_mAP_cx   = []
    for qcx in (hs.test_sample_cx):
        res = qcx2_res[qcx]
        mAP = get_oxsty_mAP_score_from_res(hs, res, SV, oxsty_qres_dpath)
        query_mAP_list.append(mAP)
        query_mAP_cx.append(qcx)
    # Calculate the total mAP score for the experiemnt
    total_mAP = np.mean(np.array(query_mAP_list))
    # build a CSV file with the results
    header  = '# Oxford Style Map Scores'
    header  = '# total mAP score = %r ' % total_mAP
    header +=  get_timestamp(format='comment')+'\n'
    header += '# Full Parameters: \n#' + params.param_string().replace('\n','\n#')+'\n\n'
    column_labels = ['QCX', 'mAP']
    column_list   = [query_mAP_cx, query_mAP_list]
    oxsty_map_csv = load_data2.make_csv_table(column_labels, column_list, header)
    return oxsty_map_csv

def get_oxsty_mAP_score_from_res(hs, res, SV, oxsty_qres_dpath):
    # find oxford ground truth directory
    cwd = os.getcwd()
    oxford_gt_dir = join(hs.dirs.db_dir, 'oxford_style_gt')
    # build groundtruth query
    qcx = res.qcx
    qnx = hs.tables.cx2_nx[qcx]
    cx2_oxnum = hs.tables.prop_dict['oxnum']
    qoxnum = cx2_oxnum[qcx]
    qname  = hs.tables.nx2_name[qnx]
    # build ranked list
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    top_cx = cx2_score.argsort()[::-1]
    top_gx = hs.tables.cx2_gx[top_cx]
    top_gname = hs.tables.gx2_gname[top_gx]
    # build mAP args
    ground_truth_query = qname+'_'+qoxnum
    # build ranked list of gnames (remove duplicates)
    seen = set([])
    ranked_list = []
    for gname in iter(top_gname):
        gname_ = gname.replace('.jpg','')
        if not gname_ in seen: 
            seen.add(gname_)
            ranked_list.append(gname_)
    ranked_list2 = [gname.replace('.jpg','') for gname in top_gname]
    # Write the ranked list of images names
    cx_aug = 'qcx_'+str(qcx)
    ranked_list_fname = 'ranked_list_' + cx_aug + ground_truth_query + '.txt'
    ranked_list_fpath = join(oxsty_qres_dpath, ranked_list_fname)
    helpers.write_to(ranked_list_fpath, '\n'.join(ranked_list))
    # execute external mAP code: 
    # ./compute_ap [GROUND_TRUTH] [RANKED_LIST]
    os.chdir(oxford_gt_dir)
    args = ('../compute_ap', ground_truth_query, ranked_list_fpath)
    cmdstr  = ' '.join(args)
    print('Executing: %r ' % cmdstr)
    PIPE = subprocess.PIPE
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
    (out, err) = proc.communicate()
    return_code = proc.returncode
    os.chdir(cwd)
    mAP = float(out.strip())
    return mAP

# NEW STUFF
def get_top_matches_cx_and_scores(hs, res, SV):
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    qcx = res.qcx
    # get top chip indexes which were in the database
    db_sample_cx = range(len(cx2_desc)) if hs.database_sample_cx is None \
                               else hs.database_sample_cx
    unfilt_top_cx = np.argsort(cx2_score)[::-1]
    top_cx = np.array(helpers.intersect_ordered(unfilt_top_cx, db_sample_cx))
    top_score = cx2_score[top_cx]
    return top_cx, top_score

def __get_top_matches_true_and_false(hs, res, top_cx, top_score):
    qcx    = res.qcx
    top_nx = hs.tables.cx2_nx[top_cx]
    qnx    = hs.tables.cx2_nx[qcx]
    true_ranks  = np.where(np.logical_and(top_nx == qnx, top_cx != qcx))[0]
    false_ranks = np.where(np.logical_and(top_nx != qnx, top_cx != qcx))[0]
    return true_ranks, false_ranks

def get_top_matches_true_and_false(hs, res, SV):
    top_cx, top_score = get_top_matches_cx_and_scores(hs, res, sv)
    return __get_top_matches_true_and_false(hs, res, top_cx, top_score)

def get_matchs_true_and_false(hs, res, SV):
    top_cx, top_score = get_top_matches_cx_and_scores(hs, res, SV)
    true_ranks, false_ranks = __get_top_matches_true_and_false(hs, res, top_cx, top_score)
    # Get true score / cx
    true_scores = top_score[true_ranks]
    true_cxs    = top_cx[true_ranks]
    # Get false score / cx
    false_scores = top_score[false_ranks]
    false_cxs    = top_cx[false_ranks]
    # Put them in tuple and return
    true_tup  = (true_cxs, true_scores, true_ranks)
    false_tup = (false_cxs, false_scores, false_ranks)
    return true_tup, false_tup

# OLD STUFF
def get_tt_bt_tf_cxs(hs, res, SV):
    'Returns the top and bottom true positives and top false positive'
    qcx = res.qcx
    true_tup, false_tup = get_matchs_true_and_false(hs, res, SV)
    true_cxs,  true_scores,  true_ranks  = true_tup
    false_cxs, false_scores, false_ranks = false_tup
    nth_true = lambda n: (true_cxs[n], true_ranks[n], true_scores[n])
    nth_false = lambda n: (false_cxs[n], false_ranks[n], false_scores[n])
    if len(true_ranks) == 0:
        tt_cx, tt_r, tt_s = (np.nan, np.nan, np.nan)
        bt_cx, bt_r, bt_s = (np.nan, np.nan, np.nan)
    else:
        tt_cx, tt_r, tt_s = nth_true(0)
        bt_cx, bt_r, bt_s = nth_true(-1)
    tf_cx, tf_r, tf_s = nth_false(0)

    titles = ('best True rank='+str(tt_r)+' ',
              'worst True rank='+str(bt_r)+' ',
              'best False rank='+str(tf_r)+' ')
    cxs = (tt_cx, bt_cx, tf_cx)
    return cxs, titles

def visualize_res_tt_bt_tf(hs, res):
    #print('Visualizing result: ')
    #res.printme()
    SV = False
    qcx = res.qcx
    _fn = qcx
    cxs, titles = get_tt_bt_tf_cxs(hs, res, SV)
    df2.show_matches3(res, hs, cxs[0], SV, fignum=_fn, plotnum=231, title_aug=titles[0])
    df2.show_matches3(res, hs, cxs[1], SV, fignum=_fn, plotnum=232, title_aug=titles[1])
    df2.show_matches3(res, hs, cxs[2], SV, fignum=_fn, plotnum=233, title_aug=titles[2])
    SV = True
    cxsV, titlesV = get_tt_bt_tf_cxs(hs, res, SV)
    df2.show_matches3(res, hs, cxsV[0], SV, fignum=_fn, plotnum=234, title_aug=titlesV[0])
    df2.show_matches3(res, hs, cxsV[1], SV, fignum=_fn, plotnum=235, title_aug=titlesV[1])
    df2.show_matches3(res, hs, cxsV[2], SV, fignum=_fn, plotnum=236, title_aug=titlesV[2])
    fig_title = 'fig '+str(_fn)+' -- ' + params.get_query_uid()
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)
    return _fn, fig_title

def visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx):
    res = qcx2_res[qcx]
    return visualize_res_tt_bt_tf(hs, res)

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
    hs = load_data2.HotSpotter(db_dir)
    df2.close_all_figures()
    try:
        qcx2_res = mc2.run_matching(hs)
        qcx = 1
        #print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=False)
        #print_top_qcx_scores(hs, qcx2_res, qcx, view_top=10, SV=True)
        if '--dump' in sys.argv:
            dump_qcx_tt_bt_tf(hs, qcx2_res)
        #visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx)

        def dinspect(qcx):
            df2.close_all_figures()
            visuzlize_qcx_tt_bt_tf(hs, qcx2_res, qcx)
            df2.present()

    except Exception as ex:
        print(repr(ex))
        raise

    # Execing df2.present does an IPython aware plt.show()
    exec(df2.present(wh=(900,600)))
