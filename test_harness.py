from __future__ import division, print_function
import itertools
import textwrap
import match_chips3 as mc3
import DataStructures as ds
import dev
import draw_func2 as df2
import fileio as io
#from match_chips3 import *
import helpers as helpers
import numpy as np
import sys
from os.path import join
import _test_configurations as _testcfgs

# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed

# What happens when we take all other possible ground truth matches out
# of the database index?

#mc3.rrr(); mf.rrr(); dev.rrr(); ds.rrr(); vr2.rrr()
#vr2.print_off()
#mf.print_off()
#algos.print_off()
#io.print_off()
#HotSpotter.print_off()
#cc2.print_off()
#fc2.print_off()
#ld2.print_off()
#helpers.print_off()
#parallel.print_off()
#mc3.print_off()


def get_valid_testcfg_names():
    testcfg_keys = vars(_testcfgs).keys()
    testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    valid_cfg_names = helpers.indent('\n'.join(testcfg_locals), '  * ')
    return valid_cfg_names


def get_vary_dicts(test_cfg_name_list):
    vary_dicts = []
    for cfg_name in test_cfg_name_list:
        evalstr = '_testcfgs.' + cfg_name
        test_cfg = eval(evalstr)
        vary_dicts.append(test_cfg)
    if len(vary_dicts) == 0:
        valid_cfg_names = get_valid_testcfg_names()
        raise Exception('Choose a valid testcfg:\n' + valid_cfg_names)
    return vary_dicts


#---------
# Helpers
#---------------
# Display Test Results
def print_test_results(test_results):
    print('[harn] ---')
    (col_lbls, row_lbls, mat_vals, test_uid, nLeX) = test_results
    test_uid = mc3.simplify_test_uid(test_uid)
    print('[harn] test_uid=%r' % test_uid)
    #print('[harn] row_lbls=\n%s' % str(row_lbls))
    #print('[harn] col_lbls=\n%s' % str('\n  '.join(col_lbls)))
    print('[harn] lowest_gt_ranks(NN,FILT,SV)=\n%s' % str(mat_vals))


#---------------
# Display Test Results
#-----------
# Run configuration for each qon
def get_test_results(hs, qon_list, query_cfg, cfgx=0, nCfg=1,
                     force_load=False):
    print('[harn] get_test_results(): %r' % query_cfg.get_uid())
    query_uid = query_cfg.get_uid()
    hs_uid    = hs.db_name()
    qon_uid   = helpers.hashstr(repr(tuple(qon_list)))
    test_uid  = hs_uid + query_uid + qon_uid
    cache_dir = join(hs.dirs.cache_dir, 'test_harness_results')
    io_kwargs = dict(dpath=cache_dir, fname='test_results', uid=test_uid, ext='.cPkl')
    # High level caching
    qonx2_bestranks = []
    #nChips = hs.num_cx
    #nNames = len(hs.tables.nx2_name) - 2
    nQuery = len(qon_list)
    #NMultiNames =
    nPrevQ = nQuery * cfgx
    qonx2_reslist = []
    if  not hs.args.nocache_query and (not force_load):
        test_results = io.smart_load(**io_kwargs)
        if test_results is None:
            pass
        elif len(test_results) != 1:
            print('recaching test_results')
        elif not test_results is None:
            return test_results, [[{0: None}]] * nQuery
    for qonx, (qcx, ocids, notes) in enumerate(qon_list):
        print(textwrap.dedent('''
        [harn]----------------
        [harn] TEST %d/%d
        [harn]----------------''' % (qonx + nPrevQ + 1, nQuery * nCfg)))
        gt_cxs = hs.get_other_indexed_cxs(qcx)
        #title = 'q' + hs.cidstr(qcx) + ' - ' + notes
        #print('[harn] title=%r' % (title,))
        #print('[harn] gt_' + hs.cidstr(gt_cxs))
        res_list = mc3.execute_query_safe(hs, query_cfg, [qcx])
        bestranks = []
        algos = []
        qonx2_reslist += [res_list]
        assert len(res_list) == 1
        for qcx2_res in res_list:
            assert len(qcx2_res) == 1
            res = qcx2_res[qcx]
            algos += [res.title]
            gt_ranks = res.get_gt_ranks(gt_cxs)
            #print('[harn] cx_ranks(/%4r) = %r' % (nChips, gt_ranks))
            #print('[harn] cx_ranks(/%4r) = %r' % (NMultiNames, gt_ranks))
            #print('ns_ranks(/%4r) = %r' % (nNames, gt_ranks))
            if len(gt_ranks) == 0:
                _bestrank = -1
            else:
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
def test_configurations(hs, qon_list, test_cfg_name_list, fnum=1):
    vary_dicts = get_vary_dicts(test_cfg_name_list)
    print('\n*********************\n')
    print('[harn]================')
    print('[harn]test_scoring(hs)')
    #vary_dicts = vary_dicts[0]
    varied_params_list = [_ for _dict in vary_dicts for _ in helpers.all_dict_combinations(_dict)]
    # query_cxs, other_cxs, notes
    qon_list = dev.get_qon_list(hs)
    cfg_list = [ds.QueryConfig(hs, **_dict) for _dict in varied_params_list]
    # __NEW_HACK__
    mc3.unify_cfgs(cfg_list)
    # __END_HACK__
    # Preallocate test result aggregation structures
    print('')
    print('[harn] Testing %d different parameters' % len(cfg_list))
    print('[harn]         %d different chips' % len(qon_list))
    nCfg = len(cfg_list)
    nQuery = len(qon_list)
    rc2_res = np.empty((nQuery, nCfg), dtype=list)
    mat_list = []
    for cfgx, test_cfg in enumerate(cfg_list):
        print(textwrap.dedent('''
        [harn]---------------')
        [harn] TEST_CFG %d/%d'
        [harn]---------------'''  % (cfgx + 1, nCfg)))
        force_load = cfgx in hs.args.c
        (mat_vals, ), qonx2_reslist = get_test_results(hs, qon_list, test_cfg,
                                                       cfgx, nCfg, force_load)
        mat_list.append(mat_vals)
        for qonx, reslist in enumerate(qonx2_reslist):
            assert len(reslist) == 1
            qcx2_res = reslist[0]
            assert len(qcx2_res) == 1
            res = qcx2_res.values()[0]
            rc2_res[qonx, cfgx] = res
        # Keep the best results
    print('[harn] Finished testing parameters')
    print('')
    print('---------------------------------')
    #--------------------
    # Print Best Results
    rank_mat = np.hstack(mat_list)
    # Label the rank matrix:
    _colxs = np.arange(nCfg)
    lbld_mat = np.vstack([_colxs, rank_mat])
    _rowxs = np.arange(nQuery + 1).reshape(nQuery + 1, 1) - 1
    lbld_mat = np.hstack([_rowxs, lbld_mat])
    # Build row labels
    qonx2_lbl = []
    for qonx in xrange(nQuery):
        qcx, ocxs, notes = qon_list[qonx]
        label = 'qonx %d) q%s -- notes=%s' % (qonx, hs.cidstr(qcx), notes)
        qonx2_lbl.append(label)
    qonx2_lbl = np.array(qonx2_lbl)
    # Build col labels
    cfgx2_lbl = []
    for cfgx in xrange(nCfg):
        test_uid  = mc3.simplify_test_uid(cfg_list[cfgx].get_uid())
        test_uid  = mc3.simplify_test_uid(cfg_list[cfgx].get_uid())
        cfg_label = 'cfgx %3d) %s' % (cfgx, test_uid)
        cfgx2_lbl.append(cfg_label)
    cfgx2_lbl = np.array(cfgx2_lbl)
    #------------
    indent = helpers.indent
    print('')
    print('[harn]-------------')
    print('[harn] queries:\n%s' % '\n'.join(qonx2_lbl))
    #------------
    print('')
    print('[harn]-------------')
    print('[harn] configs:\n%s' % '\n'.join(cfgx2_lbl))
    #------------
    PRINT_ROW_SCORES = True and (not '--noprintrow' in sys.argv)
    if PRINT_ROW_SCORES:
        print('')
        print('[harn]-------------')
        print('[harn] Scores per query')
        print('[harn]-------------')
        qonx2_min_rank = []
        qonx2_argmin_rank = []
        indent = helpers.indent
        new_hard_qonx_list = []
        for qonx in xrange(nQuery):
            ranks = rank_mat[qonx]
            min_rank = ranks.min()
            bestCFG_X = np.where(ranks == min_rank)[0]
            qonx2_min_rank.append(min_rank)
            qonx2_argmin_rank.append(bestCFG_X)
            print('[row_score] %3d) %s' % (qonx, qonx2_lbl[qonx]))
            print('[row_score] best_rank = %d ' % min_rank)
            print('[row_score] minimizing_configs = %s ' %
                  indent('\n'.join(cfgx2_lbl[bestCFG_X]), '    '))
            if ranks.max() > 0:
                new_hard_qonx_list += [qonx]
        print('--- hard qon_list (w.r.t these configs) ---')
        new_hard_qon_list = []
        for qonx in new_hard_qonx_list:
            # New list is in cid format instead of cx format
            # because you should be copying and pasting it
            qcx, ocxs, notes = qon_list[qonx]
            notes += ' ranks = ' + str(rank_mat[qonx])
            qcid = hs.tables.cx2_cid[qcx]
            ocids = hs.tables.cx2_cid[ocxs]
            new_hard_qon_list += [(qcid, list(ocids), notes)]
        print('\n'.join(map(repr, new_hard_qon_list)))

    #------------
    def rankscore_str(thresh, nLess, total):
        #helper to print rank scores of configs
        percent = 100 * nLess / total
        return '#ranks < %d = %d/%d = (%.1f%%) (err=%d)' % (thresh, nLess, total, percent, (total - nLess))
    print('')
    print('[harn]-------------')
    print('[harn] Scores per config')
    print('[harn]-------------')
    X_list = [1, 5]
    # Build a dictionary mapping X (as in #ranks < X) to a list of cfg scores
    nLessX_dict = {int(X): np.zeros(nCfg) for X in iter(X_list)}
    for cfgx in xrange(nCfg):
        ranks = rank_mat[:, cfgx]
        print('[col_score] %d) %s' % (cfgx, cfgx2_lbl[cfgx]))
        for X in iter(X_list):
            nLessX_ = sum(np.bitwise_and(ranks < X, ranks >= 0))
            print('[col_score] ' + rankscore_str(X, nLessX_, nQuery))
            nLessX_dict[int(X)][cfgx] = nLessX_

    LATEX_SUMMARY = True
    if LATEX_SUMMARY:
        print('--- LaTeX ---')
        # Create configuration latex table
        criteria_lbls = ['#ranks < %d' % X for X in X_list]
        db_name = hs.db_name(True)
        cfg_score_title = db_name + ' rank scores'
        cfgscores = np.array([nLessX_dict[int(X)] for X in X_list]).T
        import latex_formater as latex

        replace_rowlbl = [(' *cfgx *', ' ')]
        tabular_kwargs = dict(title=cfg_score_title, out_of=nQuery,
                              bold_best=True, replace_rowlbl=replace_rowlbl,
                              flip=True)
        tabular_str = latex.make_score_tabular(cfgx2_lbl, criteria_lbls,
                                               cfgscores, **tabular_kwargs)
        print(tabular_str)
        print('--- /LaTeX ---')
    #------------
    print('')
    print('[harn]---------------')
    print('[harn] Best configurations')
    print('[harn]---------------')
    best_rankscore_summary = []
    for X, cfgx2_nLessX in nLessX_dict.iteritems():
        max_LessX = cfgx2_nLessX.max()
        bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
        best_rankscore = '[best_cfg] %d config(s) scored ' % len(bestCFG_X)
        best_rankscore += rankscore_str(X, max_LessX, nQuery)
        best_rankcfg = indent('\n'.join(cfgx2_lbl[bestCFG_X]), '    ')
        print(best_rankscore)
        print(best_rankcfg)
        best_rankscore_summary += [best_rankscore]
    #------------
    PRINT_MAT = True and (not '--noprintmat' in sys.argv)
    if PRINT_MAT:
        print('')
        print('[harn]-------------')
        print('[harn] labled rank matrix: rows=queries, cols=cfgs:\n%s' % lbld_mat)
        print('[harn]-------------')
    #------------
    print('[col_score] --- summary ---')
    print('\n'.join(best_rankscore_summary))
    # Draw results
    print(hs.args.r)
    for r, c in itertools.product(hs.args.r, hs.args.c):
        #print('viewing (r,c)=(%r,%r)' % (r,c))
        res = rc2_res[r, c]
        #res.printme()
        res.show_topN(hs, fignum=fnum)
        fnum += 1
    print('--remember you have --r and --c available to you')

if __name__ == '__main__':
    import multiprocessing
    np.set_printoptions(threshold=5000, linewidth=5000)
    multiprocessing.freeze_support()
    print('[harn]-----------')
    print('[harn] main()')
    main_locals = dev.dev_main()
    hs = main_locals['hs']
    qon_list = main_locals['qon_list']
    #test_cfg_name_list = ['vsone_1']
    #test_cfg_name_list = ['vsmany_3456']
    test_cfg_name_list = ['vsmany_srule']
    test_configurations(hs, qon_list, test_cfg_name_list)
    exec(df2.present())
