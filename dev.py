from __future__ import division, print_function
import itertools
import match_chips3 as mc3
import matching_functions as mf
import DataStructures as ds
import investigate_chip as iv
import voting_rules2 as vr2
import draw_func2 as df2
import fileio as io
import chip_compute2 as cc2
import feature_compute2 as fc2
import load_data2 as ld2
import HotSpotter
import algos
import re
import Parallelize as parallel
#from match_chips3 import *
import helpers as helpers
import numpy as np
import sys
from os.path import join
from collections import OrderedDict

# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed

# What happens when we take all other possible ground truth matches out
# of the database index? 

#mc3.rrr(); mf.rrr(); iv.rrr(); ds.rrr(); vr2.rrr()
#vr2.print_off()
#mf.print_off()
#algos.print_off()
#io.print_off()
HotSpotter.print_off()
#cc2.print_off()
#fc2.print_off()
#ld2.print_off()
#helpers.print_off()
#parallel.print_off()
#mc3.print_off()

def get_vary_dicts(args):
    import _test_configurations as _testcfgs
    vary_dicts = []
    if args.test_vsmany:
        #vary_dicts.append(_testcfgs.vsmany_3456)
        vary_dicts.append(_testcfgs.vsmany_2)
    if args.test_vsone:
        vary_dicts.append(_testcfgs.vsone)
    if len(vary_dicts) == 0: 
        raise Exception('choose --test-vsmany')
    return vary_dicts

#---------
# Helpers
def update_test_results(qonx2_agg, test_results):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, 
     qonx2_best_col, qonx2_score, mats_list) = qonx2_agg
    (col_lbls, row_lbls, mat_vals, test_uid, nLeX) = test_results
    test_uid = simplify_test_uid(test_uid)
    best_vals = mat_vals.min(1)
    best_mat = np.tile(best_vals, (len(mat_vals.T), 1)).T
    best_pos = (best_mat == mat_vals)
    mats_list += [mat_vals]
    for qonx, val in enumerate(best_vals):
        if val == qonx2_score[qonx]:
            colpos = np.where(best_pos[qonx])[0].min()
            qonx2_best_params[qonx] += '\n'+test_uid+'c'+str(colpos)
        if val < qonx2_score[qonx]  or qonx2_score[qonx] == -1:
            qonx2_score[qonx] = val
            colpos = np.where(best_pos[qonx])[0].min()
            qonx2_best_params[qonx] = test_uid+'c'+str(colpos)
            qonx2_lbl[qonx] = row_lbls[qonx]
            qonx2_colpos[qonx] = colpos

def simplify_test_uid(test_uid):
    # Remove extranious characters from test_uid
    test_uid = re.sub(r'_trainID\([0-9]*,........\)','', test_uid)
    test_uid = re.sub(r'_indxID\([0-9]*,........\)','', test_uid)
    test_uid = re.sub(r'HSDB_zebra_with_mothers','', test_uid)
    test_uid = re.sub(r'GZ_ALL','', test_uid)
    test_uid = re.sub(r'HESAFF_sz750','', test_uid)
    test_uid = test_uid.strip(' _')
    return test_uid

#---------------
# Display Test Results
def print_test_results(test_results):
    print('[dev] ---')
    (col_lbls, row_lbls, mat_vals, test_uid, nLeX) = test_results
    test_uid = simplify_test_uid(test_uid)
    print('[dev] test_uid=%r' % test_uid)
    #print('[dev] row_lbls=\n%s' % str(row_lbls))
    #print('[dev] col_lbls=\n%s' % str('\n  '.join(col_lbls)))
    print('[dev] lowest_gt_ranks(NN,FILT,SV)=\n%s' % str(mat_vals))

#---------------
# Display Test Results
def print_best(qonx2_agg, cfg_list):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, 
     qonx2_best_col, qonx2_score, mats_list) = qonx2_agg
    print('')
    print('--------------------------------------------')
    print('[best_qon] printing best results over %d queries' % (len(qonx2_lbl)))
    for row in xrange(len(qonx2_lbl)):
        best_params_str = helpers.indent('\n'+qonx2_best_params[row], '    ')
        print('[best_qon] --- %r/%r ----' % (row+1, len(qonx2_lbl)))
        print('[best_qon] rowlbl(%d): %s' % (row, qonx2_lbl[row]))
        print('[best_qon] best_params = %s' % (best_params_str,))
        print('[best_qon] best_score(c%r) = %r' % (qonx2_colpos[row], qonx2_score[row],))
    print('[best_qon] ---- END ----')
    _2str = lambda cfgx, cfg: ('%3d) ' % cfgx)+simplify_test_uid(cfg.get_uid())
    rowlbl_list = [('%3d) ' % qonx)+str(lbl) for qonx, lbl in enumerate(qonx2_lbl)]
    collbl_list = [_2str(*tup) for tup in enumerate(cfg_list)]
    print('[best_all] Row Labels: ')
    print('    '+'\n    '.join(rowlbl_list))
    print('[best_all] Column Labels: ')
    print('    '+'\n    '.join(collbl_list))
    rank_mat = np.hstack(mats_list)
    rank_mat = np.vstack([np.arange(rank_mat.shape[1]), rank_mat])
    rank_mat = np.hstack([np.arange(rank_mat.shape[0]).reshape(rank_mat.shape[0],1)-1, rank_mat])
    print('[best_all] all_ranks (rows=chip, cols=cfg) = \n%s' % str(rank_mat))
    qonx2_score.shape = (len(qonx2_score),1)
    #print('[best_all] best_ranks =\n%s ' % str(qonx2_score))

    print('[best_all] Finished printing best results')
    print('------------------------------------')

#-----------
# Run configuration for each qon
def get_test_results(hs, qon_list, q_params, cfgx=0, nCfg=1,
                     force_load=False):
    print('[dev] get_test_results(): %r' % q_params.get_uid())
    query_uid = q_params.get_uid()
    hs_uid    = hs.db_name()
    qon_uid   = helpers.hashstr(repr(tuple(qon_list)))
    test_uid  = hs_uid + query_uid + qon_uid
    cache_dir = join(hs.dirs.cache_dir, 'dev_test_results')
    io_kwargs = dict(dpath=cache_dir, fname='test_results', uid=test_uid, ext='.cPkl')
    # High level caching
    qonx2_bestranks = []
    nChips = hs.num_cx
    nNames = len(hs.tables.nx2_name) - 2
    nQuery = len(qon_list) 
    nPrevQ = nQuery*cfgx
    qonx2_reslist = []
    if  not hs.args.nocache_query and (not force_load):
        test_results = io.smart_load(**io_kwargs)
        if test_results is None: pass
        elif len(test_results) != 1: print('recaching test_results')
        elif not test_results is None: return test_results, [[{0:None}]]*nQuery
    for qonx, (qcx, ocids, notes) in enumerate(qon_list):
        print('[dev]----------------')
        print('[dev] TEST %d/%d' % (qonx+nPrevQ+1, nQuery*nCfg))
        print('[dev]----------------')
        gt_cxs = hs.get_other_cxs(qcx)
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        #print('[dev] title=%r' % (title,))
        #print('[dev] gt_'+hs.cxstr(gt_cxs))
        res_list = mc3.execute_query_safe(hs, q_params, [qcx])
        bestranks = []
        algos = []
        qonx2_reslist += [res_list]
        assert len(res_list) == 1
        for qcx2_res in res_list:
            assert len(qcx2_res) == 1
            res = qcx2_res[qcx]
            algos += [res.title]
            gt_ranks = res.get_gt_ranks(gt_cxs)
            print('[dev] cx_ranks(/%4r) = %r' % (nChips, gt_ranks))
            print('ns_ranks(/%4r) = %r' % (nNames, gt_ranks))
            _bestrank = min(gt_ranks)
            bestranks += [_bestrank]
        # record metadata
        qonx2_bestranks += [bestranks]
    mat_vals = np.array(qonx2_bestranks)
    test_results = (mat_vals,)
    # High level caching
    helpers.ensuredir('results')
    io.smart_save(test_results, **io_kwargs)
    return test_results, qonx2_reslist

#-----------
# Test Each configuration 
def test_configurations(hs):
    print('\n*********************\n')
    print('[dev]================')
    print('[dev]test_scoring(hs)')
    #vary_dicts = vary_dicts[0]
    vary_dicts = get_vary_dicts(hs.args)
    varied_params_list = [_ for _dict in vary_dicts for _ in helpers.all_dict_combinations(_dict)]
    # query_cxs, other_cxs, notes
    qon_list = iv.get_qon_list(hs)
    cfg_list = [ds.QueryConfig(**_dict) for _dict in varied_params_list]
    # __NEW_HACK__
    # Super HACK so all query configs share the same nearest neighbor indexes
    GLOBAL_dcxs2_index = cfg_list[0].dcxs2_index
    for q_cfg in cfg_list:
        q_cfg.dcxs2_index = GLOBAL_dcxs2_index
    # __END_HACK__
    # Preallocate test result aggregation structures
    print('')
    print('[dev] Testing %d different parameters' % len(cfg_list))
    print('[dev]         %d different chips' % len(qon_list))
    nCfg = len(cfg_list)
    nQuery = len(qon_list)
    rc2_res = np.empty((nQuery, nCfg), dtype=list)
    mat_list = []
    for cfgx, test_cfg in enumerate(cfg_list):
        print('[dev]---------------')
        print('[dev] TEST_CFG %d/%d' % (cfgx+1, nCfg))
        print('[dev]---------------')
        force_load = cfgx in hs.args.c 
        (mat_vals,), qonx2_reslist =\
                get_test_results(hs, qon_list, test_cfg, cfgx, nCfg, force_load)
        mat_list.append(mat_vals)
        for qonx, reslist in enumerate(qonx2_reslist):
            assert len(reslist) == 1
            qcx2_res = reslist[0]
            assert len(qcx2_res) == 1
            res = qcx2_res.values()[0]
            rc2_res[qonx, cfgx] = res
        # Keep the best results 
    print('[dev] Finished testing parameters')
    print('')
    print('---------------------------------')
    #--------------------
    # Print Best Results
    rank_mat = np.hstack(mat_list)
    # Label the rank matrix: 
    _colxs = np.arange(nCfg)
    lbld_mat = np.vstack([_colxs, rank_mat])
    _rowxs = np.arange(nQuery+1).reshape(nQuery+1,1)-1
    lbld_mat = np.hstack([_rowxs, lbld_mat])
    # Build row labels
    qonx2_lbl = []
    for qonx in xrange(nQuery):
        qcx, ocxs, notes = qon_list[qonx]
        label = 'qonx %d) q%s -- notes=%s' % (qonx, hs.cxstr(qcx), notes)
        qonx2_lbl.append(label)
    qonx2_lbl = np.array(qonx2_lbl)
    # Build col labels
    cfgx2_lbl = []
    for cfgx in xrange(nCfg):
        test_uid  = simplify_test_uid(cfg_list[cfgx].get_uid())
        cfg_label = 'cfgx %3d) %s' % (cfgx, test_uid)
        cfgx2_lbl.append(cfg_label)
    cfgx2_lbl = np.array(cfgx2_lbl)
    #------------
    indent = helpers.indent
    print('')
    print('[dev]-------------')
    print('[dev] queries:\n%s' % '\n'.join(qonx2_lbl))
    #------------
    print('')
    print('[dev]-------------')
    print('[dev] configs:\n%s' % '\n'.join(cfgx2_lbl))
    #------------
    print('')
    print('[dev]-------------')
    print('[dev] Scores per query')
    print('[dev]-------------')
    qonx2_min_rank = []
    qonx2_argmin_rank = []
    indent = helpers.indent
    for qonx in xrange(nQuery):
        ranks = rank_mat[qonx]
        min_rank = ranks.min()
        cfgx_list = np.where(ranks == min_rank)[0]
        qonx2_min_rank.append(min_rank)
        qonx2_argmin_rank.append(cfgx_list)
        print('[row_score] %3d) %s' % (qonx,qonx2_lbl[qonx]) )
        print('[row_score] best_rank = %d ' % min_rank)
        print('[row_score] minimizing_configs = %s ' %
              indent('\n'.join(cfgx2_lbl[cfgx_list]), '    '))
    #------------
    print('')
    print('[dev]-------------')
    print('[dev] Scores per config')
    print('[dev]-------------')
    X_list = [5, 1]
    nLessX_dict = {}
    for X in iter(X_list):
        cfgx2_nLessX = []
        for cfgx in xrange(nCfg):
            ranks = rank_mat[:,cfgx]
            nLessX_ = sum(ranks < X)
            cfgx2_nLessX.append(nLessX_)
            print(len(cfgx2_lbl))
            print('[col_score] %3d) %s' % (cfgx, cfgx2_lbl[cfgx]) )
            print('[col_score] #ranks<%d = %d ' % (X, nLessX_))
        nLessX_dict[int(X)] = np.array(cfgx2_nLessX)
    #------------
    print('')
    print('[dev]---------------')
    print('[dev] Best configurations')
    print('[dev]---------------')
    for X, cfgx2_nLessX in nLessX_dict.iteritems():
        min_LessX = cfgx2_nLessX.max()
        bestCFG_X = np.where(cfgx2_nLessX == min_LessX)
        print('[best_cfg]%d config(s) scored #ranks<%d = %d/%d' % \
              (len(bestCFG_X), X, min_LessX, nQuery))
        print(indent('\n'.join(cfgx2_lbl[cfgx_list]), '    '))
    #------------
    print('')
    print('[dev]-------------')
    print('[dev] labled rank matrix: rows=queries, cols=cfgs:\n%s' % lbld_mat)
    print('[dev]-------------')
    #------------
    # Draw results
    fnum = 0
    print(hs.args.r)
    for r,c in itertools.product(hs.args.r, hs.args.c):
        print('viewing (r,c)=(%r,%r)' % (r,c))
        res = rc2_res[r,c]
        res.printme()
        res.show_topN(hs, fignum=fnum)
        fnum += 1

if __name__ == '__main__':
    import multiprocessing
    np.set_printoptions(threshold=5000, linewidth=5000)
    multiprocessing.freeze_support()
    print('[dev]-----------')
    print('[dev] main()')
    df2.DARKEN = .5
    main_locals = iv.main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    test_configurations(hs)
    exec(df2.present())
