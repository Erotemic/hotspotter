# Hotspotter imports
import helpers
import load_data2
import oxsty_results
import params
from Printable import DynStruct
# Scientific imports
import numpy as np
# Standard library imports
import datetime
import os
import subprocess
import sys
import textwrap
from itertools import izip
from os.path import realpath, join

# OXFORD STUFF
def write_mAP_results(allres):
    hs = allres.hs
    qcx2_res = hs.qcx2_res
    SV = allres.SV
    oxsty_map_csv = oxsty_mAP_results(hs, qcx2_res, SV)
    __dump_report(hs, oxsty_map_csv, 'oxsty-mAP', SV)

def oxsty_mAP_results(hs, qcx2_res, SV):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    SV = allres.SV
    # Check directorys where ranked lists of images names will be put
    SV_aug = ['_SVOFF_','_SVON_'][SV] #TODO: SV should go into params
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
    header +=  helpers.get_timestamp(format='comment')+'\n'
    header += '# Full Parameters: \n#' + params.param_string().replace('\n','\n#')+'\n\n'
    column_labels = ['QCX', 'mAP']
    column_list   = [query_mAP_cx, query_mAP_list]
    oxsty_map_csv = load_data2.make_csv_table(column_labels, column_list, header)
    return oxsty_map_csv

def get_mAP_score_from_res(hs, res, SV, oxsty_qres_dpath):
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

