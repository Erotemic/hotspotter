import numpy as np
import datetime
import textwrap
import sys
from os.path import realpath, join
from helpers import ensurepath


def get_expt_type(hs, matcher):
    expt_type = '_'.join([matcher.match_type, hs.feats.feat_type])
    return expt_type

def write_report(report_str, report_type, expt_type, results_path):
    timestamp = get_timestamp()
    csv_fname    = expt_type+report_type+timestamp+'.csv'
    results_path = realpath(results_path)
    ensurepath(results_path)
    rankres_csv = join(results_path, csv_fname)
    
    write_to(rankres_csv, report_str)

def write_rank_results(cx2_res, hs, matcher, results_path='results'):
    expt_type = get_expt_type(hs, matcher)
    rankres_str = rank_results(cx2_res, hs.tables, expt_type=expt_type)
    report_type = 'rank_'
    write_report(rankres_str, report_type, expt_type, results_path)

def get_true_positive_ranks(qcx, top_cx, cx2_nx):
    'Returns the ranking of the other chips which should have scored high'
    top_nx = cx2_nx[top_cx]
    qnx    = cx2_nx[qcx]
    _truepos_ranks, = np.where(top_nx == qnx)
    truepos_ranks = _truepos_ranks[top_cx[_truepos_ranks] != qcx]
    return truepos_ranks

def rank_results(cx2_res, hs_tables, expt_type=''):
    cx2_cid  = hs_tables.cx2_cid
    cx2_nx   = hs_tables.cx2_nx
    nx2_name = hs_tables.nx2_name
    cx2_top_truepos_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_truepos_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_score         = np.zeros(len(cx2_cid)) - 100

    for qcx, qcid in enumerate(cx2_cid):
        res = cx2_res[qcx]
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
    #   TTP RANK   = top (lowest) true positive rank
    #   TTN RANK   = top (lowest) true negative rank
    #   TTP SCORE  = top true positive score
    #   SCORE DISP = disparity between top-score and top-true-positive-score
    #   NAME       = Query chip-name''').strip()

    # Build the experiemnt csv header
    rankres_csv_header = '#CID, TTP RANK, TTN RANK, TTP SCORE, TTN SCORE, SCORE DISP, NAME'

    # Build the experiment csv data lines
    todisp = np.vstack([cx2_cid,
                        cx2_top_truepos_rank,
                        cx2_top_trueneg_rank,
                        cx2_top_truepos_score,
                        cx2_top_trueneg_score,
                        cx2_score_disp, 
                        cx2_nx]).T
    rankres_csv_lines = []
    for (cid, ttpr, ttnr, ttps, ttns, sdisp, nx) in todisp:
        csv_line = ('%4d, %8.0f, %8.0f, %9.2f, %9.2f, %10.2f, %s' %\
              (cid, ttpr, ttnr, ttps, ttns, sdisp, nx2_name[nx]) )
        rankres_csv_lines.append(csv_line)

    # Build the experiment summary report
    rankres_summary  = '\n'
    rankres_summary += '# Experiment Summary: '+expt_type+'\n'
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

def write_to(filename, to_write):
    with open(filename, 'w') as file:
        file.write(to_write)

def gvim(string):
    os.system('gvim '+result_csv)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    import match_chips2 as mc2
    import load_data2
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.MOTHERS
    hs = mc2.load_hotspotter(db_dir)
    argv = set([arg.lower() for arg in sys.argv])
    if any([arg1v1.lower() in argv for arg1v1 in ['1v1','one-vs-one','ovo']]):
        cx2_res = mc2.run_one_vs_one(hs)
    if any([arg1vM.lower() in argv for arg1vM in ['1vM','one-vs-many','ovm']]):
        cx2_res = mc2.run_one_vs_many(hs)
