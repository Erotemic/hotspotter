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


def get_test_results(hs, qon_list, q_params, use_cache=True, cfgx=0, nCfg=1,
                     force_load=False):
    print('[dev] get_test_results(): %r' % q_params.get_uid())
    query_uid = q_params.get_uid()
    hs_uid    = hs.db_name()
    qon_uid   = helpers.hashstr(repr(tuple(qon_list)))
    test_uid  = hs_uid + query_uid + qon_uid
    cache_dir = join(hs.dirs.cache_dir, 'dev_test_results')
    io_kwargs = dict(dpath=cache_dir, fname='test_results', uid=test_uid, ext='.cPkl')
    # High level caching
    id2_bestranks = []
    id2_algos = []
    id2_title = []
    nChips = hs.num_cx
    nNames = len(hs.tables.nx2_name) - 2
    nQuery = len(qon_list) 
    nPrevQ = nQuery*cfgx
    qonx2_reslist = []
    if use_cache and (not force_load):
        test_results = io.smart_load(**io_kwargs)
        if not test_results is None: return test_results, [[{0:None}]]*nQuery
    for qonx, (qcx, ocids, notes) in enumerate(qon_list):
        print('[dev]----------------')
        print('[dev] TEST %d/%d' % (qonx+nPrevQ+1, nQuery*nCfg))
        print('[dev]----------------')
        gt_cxs = hs.get_other_cxs(qcx)
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        #print('[dev] title=%r' % (title,))
        #print('[dev] gt_'+hs.cxstr(gt_cxs))
        res_list = mc3.execute_query_safe(hs, q_params, [qcx], use_cache=use_cache)
        bestranks = []
        algos = []
        qonx2_reslist += [res_list]
        for qcx2_res in res_list:
            res = qcx2_res[qcx]
            algos += [res.title]
            gt_ranks = res.get_gt_ranks(gt_cxs)
            #print('[dev] cx_ranks(/%4r) = %r' % (nChips, gt_ranks))
            #print('ns_ranks(/%4r) = %r' % (nNames, gt_ranks))
            bestranks += [min(gt_ranks)]
        # record metadata
        id2_algos += [algos]
        id2_title += [title]
        id2_bestranks += [bestranks]
    id2_title     = np.array(id2_title)
    id2_bestranks = np.array(id2_bestranks)
    id2_title.shape = (len(id2_title), 1)
    col_lbls = id2_algos[0]
    row_lbls = id2_title
    mat_vals = id2_bestranks
    test_results = (col_lbls, row_lbls, mat_vals, query_uid)
    # High level caching
    if use_cache: 
        helpers.ensuredir('results')
        io.smart_save(test_results, **io_kwargs)
    return test_results, qonx2_reslist

#---------
def update_test_results(qonx2_agg, test_results):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, 
     qonx2_best_col, qonx2_score, mats_list) = qonx2_agg
    (col_lbls, row_lbls, mat_vals, test_uid) = test_results
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

def print_test_results(test_results):
    print('[dev] ---')
    (col_lbls, row_lbls, mat_vals, test_uid) = test_results
    test_uid = simplify_test_uid(test_uid)
    print('[dev] test_uid=%r' % test_uid)
    #print('[dev] row_lbls=\n%s' % str(row_lbls))
    #print('[dev] col_lbls=\n%s' % str('\n  '.join(col_lbls)))
    print('[dev] lowest_gt_ranks(NN,FILT,SV)=\n%s' % str(mat_vals))

def print_best(qonx2_agg, test_list):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, 
     qonx2_best_col, qonx2_score, mats_list) = qonx2_agg
    print('')
    print('--------------------------------------------')
    print('[best] printing best results over %d queries' % (len(qonx2_lbl)))
    for row in xrange(len(qonx2_lbl)):
        best_params_str = helpers.indent('\n'+qonx2_best_params[row], '    ')
        print('[best] --- %r/%r ----' % (row+1, len(qonx2_lbl)))
        print('[best] rowlbl(%d): %s' % (row, qonx2_lbl[row]))
        print('[best] best_params = %s' % (best_params_str,))
        print('[best] best_score(c%r) = %r' % (qonx2_colpos[row], qonx2_score[row],))
    print('[best] ---- END ----')
    _2str = lambda cfgx, cfg: ('%3d) ' % cfgx)+simplify_test_uid(cfg.get_uid())
    rowlbl_list = [('%3d) ' % qonx)+str(lbl) for qonx, lbl in enumerate(qonx2_lbl)]
    collbl_list = [_2str(*tup) for tup in enumerate(test_list)]
    print('[best] Row Labels: ')
    print('    '+'\n    '.join(rowlbl_list))
    print('[best] Column Labels: ')
    print('    '+'\n    '.join(collbl_list))
    rank_mat = np.hstack(mats_list)
    rank_mat = np.vstack([np.arange(rank_mat.shape[1]), rank_mat])
    rank_mat = np.hstack([np.arange(rank_mat.shape[0]).reshape(rank_mat.shape[0],1)-1, rank_mat])
    print('[best] all_ranks (rows=chip, cols=cfg) = \n%r' % rank_mat)
    qonx2_score.shape = (len(qonx2_score),1)
    print('[best] best_ranks =\n%r ' % qonx2_score)

    print('[best] Finished printing best results')
    print('------------------------------------')

#------

def test_scoring(hs):
    print('\n*********************\n')
    print('[dev]================')
    print('[dev]test_scoring(hs)')
    vary_dicts = []
    vary_dicts.append({
        'query_type'     : ['vsmany'],
        'checks'         : [1024],#, 8192],
        'K'              : [10], #5, 10],
        'Knorm'          : [1], #2, 3],
        'Krecip'         : [1], #, 5, 10],
        'roidist_weight' : [0], # 1,]
        'recip_weight'   : [0], # 1,] 
        'bursty_weight'  : [0], # 1,]
        'ratio_weight'   : [0], # 1,]
        'lnbnn_weight'   : [1], # 1,]
        'lnrat_weight'   : [0], # 1,]
        'roidist_thresh' : [None], # .5,] 
        'recip_thresh'   : [None], # 0
        'bursty_thresh'  : [None], #
        'ratio_thresh'   : [None], # 1.2, 1.6
        'lnbnn_thresh'   : [None], # 
        'lnrat_thresh'   : [None], #
        'nShortlist'   : [500],
        'sv_on'        : [True], #True, False],
        'score_method' : ['pl'],#, 'pl'], #, 'nsum', 'borda', 'topk', 'nunique']
        'isWeighted'   : [True], #, False
        'max_alts'     : [200],
    })
    #vary_dicts = vary_dicts[0]
    varied_params_list = [_ for _dict in vary_dicts for _ in helpers.all_dict_combinations(_dict)]
    #for _dict in varied_params_list[0]:
        #print(_dict)
    test_list = [ds.QueryConfig(**_dict) for _dict in varied_params_list]
    print(test_list)
    #io.print_off()
    # query_cxs, other_cxs, notes
    qon_list = iv.get_qon_list(hs)
    use_cache = not hs.args.nocache_query
    # Preallocate test result aggregation structures
    qonx2_best_params = ['' for _ in xrange(len(qon_list))]
    qonx2_lbl         = ['' for _ in xrange(len(qon_list))]
    qonx2_colpos      = ['' for _ in xrange(len(qon_list))]
    qonx2_best_col    = -np.ones(len(qon_list))
    qonx2_score       = -np.ones(len(qon_list))
    mats_list         = []
    qonx2_agg = (qonx2_best_params, qonx2_lbl, qonx2_colpos, 
                 qonx2_best_col, qonx2_score, mats_list)
    #
    print('')
    print('[dev] Testing %d different parameters' % len(test_list))
    print('[dev]         %d different chips' % len(qon_list))
    testx2_results = []
    nCfg = len(test_list)
    nQuery = len(qon_list)
    rowx_cfgx2_res = np.empty((nQuery, nCfg), dtype=list)
    for cfgx, test_cfg in enumerate(test_list):
        print('[dev]---------------')
        print('[dev] TEST_CFG %d/%d' % (cfgx+1, nCfg))
        print('[dev]---------------')
        force_load = cfgx in hs.args.c
        test_results, qonx2_reslist = get_test_results(hs, qon_list, test_cfg, use_cache, cfgx, nCfg, force_load)
        testx2_results.append(test_results)
        for qonx, reslist in enumerate(qonx2_reslist):
            print(qonx)
            assert len(reslist) == 1
            qcx2_res = reslist[0]
            assert len(qcx2_res) == 1
            res = qcx2_res.values()[0]
            rowx_cfgx2_res[qonx, cfgx] = res
        # Keep the best results 
        update_test_results(qonx2_agg, test_results)
    print('[dev] Finished testing parameters')
    print('')
    print('---------------------------------')
    print('[dev] printing test rank matrices')
    for test_results in iter(testx2_results):
        print_test_results(test_results)
    print_best(qonx2_agg, test_list)
    for r,c in itertools.product(hs.args.r, hs.args.c):
        print('viewing (r,c)=(%r,%r)' % (r,c))
        res = rowx_cfgx2_res[r,c]
        res.printme()
        res.show_topN(hs)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[dev]-----------')
    print('[dev] main()')
    df2.DARKEN = .5
    main_locals = iv.main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    test_scoring(hs)
    exec(df2.present())
