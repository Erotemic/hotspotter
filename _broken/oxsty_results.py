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
from os.path import realpath, join, normpath

OXSTY_VERBOSE = False

def reload_module():
    import imp, sys
    print('[oxy] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

# OXFORD STUFF
def oxsty_mAP_results(allres):
    print('oxsty_results> Building oxsty results')
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    SV = allres.SV
    # Check directorys where ranked lists of images names will be put
    oxsty_qres_dname = 'oxsty_ranked_lists' +allres.title_suffix
    oxsty_qres_dpath = join(hs.dirs.qres_dir, oxsty_qres_dname)
    helpers.ensure_path(oxsty_qres_dpath)

    oxford_gt_dir = join(hs.dirs.db_dir, 'oxford_style_gt')
    helpers.assertpath(oxford_gt_dir)
    compute_ap_exe = normpath(join(oxford_gt_dir, '../compute_ap'))
    if not helpers.checkpath(compute_ap_exe):
        compute_ap_exe = normpath(join(oxford_gt_dir, '/compute_ap'))
    helpers.assertpath(compute_ap_exe)
    # Get the mAP scores using philbins program
    query_mAP_list = []
    query_mAP_cx   = []
    for qcx in iter(hs.test_sample_cx):
        res = qcx2_res[qcx]
        mAP = get_oxsty_mAP_score_from_res(hs, res, SV, oxsty_qres_dpath,
                                           compute_ap_exe, oxford_gt_dir)
        query_mAP_list.append(mAP)
        query_mAP_cx.append(qcx)
    print('')
    # Calculate the scalar mAP score for the experiemnt
    scalar_mAP = np.mean(np.array(query_mAP_list))
    scalar_mAP_str = '# mAP score = %r\n' % scalar_mAP
    # build a CSV file with the results
    header  = '# Oxford Style Map Scores: title_suffix=%r\n' % allres.title_suffix
    header += scalar_mAP_str
    header += helpers.get_timestamp(format='comment')+'\n'
    header += '# Full Parameters: \n#' + params.param_string().replace('\n','\n#')+'\n\n'
    column_labels = ['QCX', 'mAP']
    column_list   = [query_mAP_cx, query_mAP_list]
    oxsty_map_csv = load_data2.make_csv_table(column_labels, column_list, header)
    return oxsty_map_csv, scalar_mAP_str

def get_oxsty_mAP_score_from_res(hs, res, SV, oxsty_qres_dpath,
                                 compute_ap_exe, oxford_gt_dir):
    # find oxford ground truth directory
    cwd = os.getcwd()
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
    if qoxnum == '':
        print("HACK: Adding a dummy qoxynum")
        qoxnum = '1'
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

    def filename(path):
        return os.path.split(path)[1]

    if OXSTY_VERBOSE:
        printable_cmd = ' '.join((filename(compute_ap_exe),
                                ground_truth_query,
                                filename(ranked_list_fpath)))
        print('Executing: %r' % printable_cmd)
    else:
        helpers.print_('.')
    args = (compute_ap_exe, ground_truth_query, ranked_list_fpath)

    cmdstr  = ' '.join(args)

    try:
        proc_out = run_process(args)
        out = proc_out.out
    except OSError as ex:
        out = -1
        if OXSTY_VERBOSE:
            print(repr(ex))
        if repr(ex) == "OSError(12, 'Cannot allocate memory')":
            args_hash = helpers.hashstr(args)
            proc_err_fname = 'proc_err'+args_hash
            proc_err_cmd = proc_err_fname+'.cmd'
            proc_err_out = proc_err_fname+'.out'
            helpers.write_to(proc_err_cmd, repr(args))
            if helpers.checkpath(proc_err_out):
                out = helpers.read_from(proc_err_out)
    mAP = float(out.strip())
    os.chdir(cwd)
    return mAP

from Printable import DynStruct
def run_process(args):
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    toret = DynStruct()
    (out, err) = proc.communicate()
    toret.out = out
    toret.err = err
    toret.return_code = proc.returncode
    toret.proc = proc
    return toret

def execute(cmdstr):
    import shlex
    args = shlex.split(cmdstr)

    popen_kwargs = {
    'bufsize'            :     0,
    'executable'         :  None,
    'stdin'              :  None,
    'stdout'             :  None,
    'stderr'             :  None,
    'preexec_fn'         :  None,
    'close_fds'          : False,
    'shell'              : False,
    'cwd'                :  None,
    'env'                :  None,
    'universal_newlines' : False,
    'startupinfo'        :  None,
    'creationflags'      : 0
    }

    popen_kwargs['stdout'] = subprocess.PIPE
    popen_kwargs['stderr'] = subprocess.PIPE
    popen_kwargs['bufsize'] = -1
    popen_kwargs['shell'] = True
    popen_kwargs['close_fds'] = True

    proc = subprocess.Popen(args, **popen_kwargs)
    (out, err) = proc.communicate()
    return_code = proc.returncode

def debug_compute_ap_exe(compute_ap_exe,
                         ground_truth_query,
                         ranked_list_fpath):

    print('================================')
    print('Debugging compute_ap executable:')
    print('-----------')
    print('Path checks: ')
    helpers.checkpath(ranked_list_fpath, True)
    helpers.checkpath(compute_ap_exe, True)
    print('-----------')
    print('Command string check:')
    args = (compute_ap_exe, ground_truth_query, ranked_list_fpath)
    cmdstr  = ' '.join(args)
    print(cmdstr)
    print('-----------')
    print('Noargs check:')
    (out, err, return_code) = execute(compute_ap_exe)
    (out, err, return_code) = popen_communicate(cmdstr)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
