from __future__ import division, print_function
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


def get_test_results(hs, qon_list, q_params, use_cache=True):
    print('[dev] get_test_results(): %r' % q_params.get_uid())
    query_uid = q_params.get_uid()
    hs_uid    = hs.db_name()
    test_uid  = hs_uid + query_uid
    #io_kwargs = dict(dpath='results', fname='test_results', uid=test_uid, ext='.cPkl')
    #if use_cache:
        #test_results = io.smart_load(**io_kwargs)
        #if not test_results is None: return test_results
    id2_bestranks = []
    id2_algos = []
    id2_title = []
    nChips = hs.num_cx
    nNames = len(hs.tables.nx2_name) - 2
    for qonx, (qcx, ocids, notes) in enumerate(qon_list):
        gt_cxs = hs.get_other_cxs(qcx)
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        #print('[dev] title=%r' % (title,))
        #print('[dev] gt_'+hs.cxstr(gt_cxs))
        res_list = mc3.execute_query_safe(hs, q_params, [qcx], use_cache=use_cache)
        bestranks = []
        algos = []
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
    test_results = (col_lbls, row_lbls, mat_vals, test_uid)
    #helpers.ensuredir('results')
    #io.smart_save(test_results, **io_kwargs)
    return test_results

def test_scoring(hs):
    print('\n*********************\n')
    print('[dev]================')
    print('[dev]test_scoring(hs)')
    varied_params = {
        'lnbnn_weight' : [0, 1],
        'checks'       : [128],
        'Krecip'       : [0],
        'K'            : [5],
        'score_method' : ['csum', 'placketluce'],
        'query_type'   : ['vsmany', 'vsone'],
    }
    dict_list = helpers.all_dict_combinations(varied_params)
    test_list = [ds.QueryConfig(**_dict) for _dict in dict_list]
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
    qonx2_agg = (qonx2_best_params, qonx2_lbl, qonx2_colpos, qonx2_best_col, qonx2_score)
    #
    print('\n[dev] Testing %d different parameters' % len(test_list))
    print('\n[dev]         %d different chips' % len(qon_list))
    testx2_results = []
    for testnum, test_params in enumerate(test_list):
        print('--------------')
        print('[dev] test_params %d/%d' % (testnum+1, len(test_list)))
        test_results = get_test_results(hs, qon_list, test_params, use_cache)
        testx2_results.append(test_results)
        # Keep the best results 
        update_test_results(qonx2_agg, test_results)
    print('[dev] Finished testing parameters')
    print('---------------------------------')
    print('[dev] printing test rank matrices')
    for test_results in iter(testx2_results):
        print_test_results(test_results)
    print_best(qonx2_agg)

def update_test_results(qonx2_agg, test_results):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, qonx2_best_col, qonx2_score) = qonx2_agg
    (col_lbls, row_lbls, mat_vals, test_uid) = test_results
    test_uid = simplify_test_uid(test_uid)
    best_vals = mat_vals.min(1)
    best_mat = np.tile(best_vals, (len(mat_vals.T), 1)).T
    best_pos = (best_mat == mat_vals)
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

def print_best(qonx2_agg):
    (qonx2_best_params, qonx2_lbl, qonx2_colpos, qonx2_best_col, qonx2_score) = qonx2_agg
    print('[best] printing best results over %d queries' % (len(qonx2_lbl)))
    for row in xrange(len(qonx2_lbl)):
        best_params_str = helpers.indent('\n'+qonx2_best_params[row], '    ')
        print('[best] --- %r/%r ----' % (row+1, len(qonx2_lbl)))
        print('[best] rowlbl(%d): %s' % (row, qonx2_lbl[row]))
        print('[best] best_params = %s' % (best_params_str,))
        print('[best] best_score(c%r) = %r' % (qonx2_colpos[row], qonx2_score[row],))
    print('[best] Finished printing best results')
    print('------------------------------------')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[dev]-----------')
    print('[dev] main()')
    main_locals = iv.main()
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    test_scoring(hs)
    exec(df2.present())
