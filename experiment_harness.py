from __future__ import division, print_function
# Python
import sys
import itertools
import textwrap
from os.path import join
# Scientific
import numpy as np
# Hotspotter
import experiment_configs
from hotspotter import Config
from hotspotter import DataStructures as ds
from hotspotter import match_chips3 as mc3
from hscom import fileio as io
from hscom import helpers
from hscom import latex_formater
#from match_chips3 import *
#import draw_func2 as df2
# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed

# What happens when we take all other possible ground truth matches out
# of the database index?

#mc3.rrr(); mf.rrr(); dev.rrr(); ds.rrr(); vr2.rrr()
#vr2.print_off()
#mf.print_off()
#io.print_off()
#HotSpotter.print_off()
#cc2.print_off()
#fc2.print_off()
#ld2.print_off()
#helpers.print_off()
#parallel.print_off()
#mc3.print_off()


def get_valid_testcfg_names():
    testcfg_keys = vars(experiment_configs).keys()
    testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    valid_cfg_names = helpers.indent('\n'.join(testcfg_locals), '  * ')
    return valid_cfg_names


def get_vary_dicts(test_cfg_name_list):
    vary_dicts = []
    for cfg_name in test_cfg_name_list:
        evalstr = 'experiment_configs.' + cfg_name
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


def ArgGaurdFalse(func):
    return __ArgGaurd(func, default=False)


def ArgGaurdTrue(func):
    return __ArgGaurd(func, default=True)


def __ArgGaurd(func, default=False):
    flag = func.func_name
    if flag.find('no') == 0:
        flag = flag[2:]
    flag = '--' + flag.replace('_', '-')

    def GaurdWrapper(*args, **kwargs):
        if helpers.get_flag(flag, default):
            return func(*args, **kwargs)
        else:
            print('\n~~~ %s ~~~\n' % flag)
    GaurdWrapper.func_name = func.func_name
    return GaurdWrapper


def rankscore_str(thresh, nLess, total):
    #helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(helpers.num2_sigfig(total)) + 'd'
    fmtstr = '#ranks < %d = ' + fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + ')'
    rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    return rankscore_str


#---------------
# Display Test Results
#-----------
# Run configuration for each query
def get_test_results(hs, qcx_list, qdat, cfgx=0, nCfg=1, nocache_testres=False,
                     test_results_verbosity=2):
    nQuery = len(qcx_list)
    dcxs = hs.get_indexed_sample()
    mc3.prepare_query(qdat, None, dcxs)
    query_uid = qdat.get_uid()
    hs_uid    = hs.get_db_name()
    qcxs_uid  = helpers.hashstr_arr(qcx_list, lbl='_qcxs')
    test_uid  = hs_uid + query_uid + qcxs_uid
    cache_dir = join(hs.dirs.cache_dir, 'experiment_harness_results')
    io_kwargs = dict(dpath=cache_dir, fname='test_results', uid=test_uid,
                     ext='.cPkl')

    if test_results_verbosity == 2:
        print('[harn] get_test_results(): %r' % query_uid)
    #io.print_on()

    # High level caching
    if not hs.args.nocache_query and (not nocache_testres):
        qx2_bestranks = io.smart_load(**io_kwargs)
        if qx2_bestranks is None:
            print('[harn] Cache returned None!')
        elif len(qx2_bestranks) != len(qcx_list):
            print('[harn] Re-Caching qx2_bestranks')
        elif not qx2_bestranks is None:
            return qx2_bestranks, [[{0: None}]] * nQuery
        #raise Exception('cannot be here')

    nPrevQ = nQuery * cfgx
    qx2_bestranks = []
    qx2_reslist = []

    # Make progress message
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST %d/%d
    ---------------------''')
    mark_progress = helpers.simple_progres_func(test_results_verbosity, msg, '.')
    total = nQuery * nCfg
    # Perform queries
    for qx, qcx in enumerate(qcx_list):
        count = qx + nPrevQ + 1
        mark_progress(count, total)
        res_list = mc3.execute_query_safe(hs, qdat, [qcx], dcxs)
        qx2_reslist += [res_list]
        assert len(res_list) == 1
        bestranks = []
        for qcx2_res in res_list:
            assert len(qcx2_res) == 1
            res = qcx2_res[qcx]
            gt_ranks = res.get_gt_ranks(hs=hs)
            #print('[harn] cx_ranks(/%4r) = %r' % (nChips, gt_ranks))
            #print('[harn] cx_ranks(/%4r) = %r' % (NMultiNames, gt_ranks))
            #print('ns_ranks(/%4r) = %r' % (nNames, gt_ranks))
            _bestrank = -1 if len(gt_ranks) == 0 else min(gt_ranks)
            bestranks += [_bestrank]
        # record metadata
        qx2_bestranks += [bestranks]
        if qcx % 4 == 0:
            sys.stdout.flush()
    print('')
    qx2_bestranks = np.array(qx2_bestranks)
    # High level caching
    helpers.ensuredir(cache_dir)
    io.smart_save(qx2_bestranks, **io_kwargs)

    if helpers.get_flag('--quit2'):
        print('Triggered --quit2')
        sys.exit(1)

    return qx2_bestranks, qx2_reslist


def get_varried_params_list(test_cfg_name_list):
    vary_dicts = get_vary_dicts(test_cfg_name_list)
    varied_params_list = [_ for _dict in vary_dicts for _ in helpers.all_dict_combinations(_dict)]
    return varied_params_list


def get_cfg_list(hs, test_cfg_name_list):
    if 'custom' == test_cfg_name_list:
        cfg_list = [hs.prefs.query_cfg]
        return cfg_list
    varied_params_list = get_varried_params_list(test_cfg_name_list)
    cfg_list = [Config.QueryConfig(hs, **_dict) for _dict in varied_params_list]
    return cfg_list


#-----------
def test_configurations(hs, qcx_list, test_cfg_name_list, fnum=1):

    # Test Each configuration
    print(textwrap.dedent("""
    *********************
    [harn]================
    [harn]test_scoring(hs)"""))

    hs.update_samples()

    # Grab list of algorithm configurations to test
    cfg_list = get_cfg_list(hs, test_cfg_name_list)
    print('')
    print('[harn] Testing %d different parameters' % len(cfg_list))
    print('[harn]         %d different chips' % len(qcx_list))

    # Preallocate test result aggregation structures
    sel_cols = hs.get_arg('sel_cols', [])  # FIXME
    sel_rows = hs.get_arg('sel_rows', [])  # FIXME
    nCfg     = len(cfg_list)
    nQuery   = len(qcx_list)
    rc2_res  = np.empty((nQuery, nCfg), dtype=list)  # row/col -> result
    mat_list = []
    qdat     = ds.QueryData()

    nocache_testres =  helpers.get_flag('--nocache-testres', False)

    if helpers.get_flag('--quit1'):
        print('Triggered --quit1')
        print(sel_cols)
        print(sel_rows)
        sys.exit(1)

    test_results_verbosity = 2
    test_cfg_verbosity = 2

    msg = textwrap.dedent('''
    ---------------------')
    [harn] TEST_CFG %d/%d'
    ---------------------''')
    mark_progress = helpers.simple_progres_func(test_cfg_verbosity, msg, '+')

    # Run each test configuration
    for cfgx, query_cfg in enumerate(cfg_list):
        mark_progress(cfgx + 1, nCfg)
        # Set data to the current config
        qdat.set_cfg(query_cfg)
        _nocache_testres = nocache_testres or (cfgx in sel_cols)
        # Run the test / read cache
        qx2_bestranks, qx2_reslist = get_test_results(hs, qcx_list, qdat, cfgx,
                                                      nCfg, _nocache_testres,
                                                      test_results_verbosity)
        # Store the results
        mat_list.append(qx2_bestranks)
        for qx, reslist in enumerate(qx2_reslist):
            assert len(reslist) == 1
            qcx2_res = reslist[0]
            assert len(qcx2_res) == 1
            res = qcx2_res.values()[0]
            rc2_res[qx, cfgx] = res

    print('')
    print('------')
    print('[harn] Finished testing parameters')
    print('---------------------------------')
    #--------------------
    # Print Best Results
    rank_mat = np.hstack(mat_list)
    # Label the rank matrix:
    _colxs = np.arange(nCfg)
    lbld_mat = np.vstack([_colxs, rank_mat])
    _rowxs = np.arange(nQuery + 1).reshape(nQuery + 1, 1) - 1
    lbld_mat = np.hstack([_rowxs, lbld_mat])
    #------------
    # Build row labels
    qx2_lbl = []
    for qx in xrange(nQuery):
        qcx = qcx_list[qx]
        label = 'qx=%d) q%s ' % (qx, hs.cidstr(qcx, notes=True))
        qx2_lbl.append(label)
    qx2_lbl = np.array(qx2_lbl)
    #------------
    # Build col labels
    cfgx2_lbl = []
    for cfgx in xrange(nCfg):
        test_uid  = cfg_list[cfgx].get_uid()
        test_uid  = cfg_list[cfgx].get_uid()
        cfg_label = 'cfgx=(%3d) %s' % (cfgx, test_uid)
        cfgx2_lbl.append(cfg_label)
    cfgx2_lbl = np.array(cfgx2_lbl)
    #------------
    indent = helpers.indent

    @ArgGaurdFalse
    def print_rowlbl():
        print('=====================')
        print('[harn] Row/Query Labels')
        print('=====================')
        print('[harn] queries:\n%s' % '\n'.join(qx2_lbl))
        print('--- /Row/Query Labels ---')
    print_rowlbl()

    #------------

    @ArgGaurdFalse
    def print_collbl():
        print('')
        print('=====================')
        print('[harn] Col/Config Labels')
        print('=====================')
        print('[harn] configs:\n%s' % '\n'.join(cfgx2_lbl))
        print('--- /Col/Config Labels ---')
    print_collbl()

    #------------
    # Build Colscore
    qx2_min_rank = []
    qx2_argmin_rank = []
    new_hard_qx_list = []
    for qx in xrange(nQuery):
        ranks = rank_mat[qx]
        min_rank = ranks.min()
        bestCFG_X = np.where(ranks == min_rank)[0]
        qx2_min_rank.append(min_rank)
        qx2_argmin_rank.append(bestCFG_X)
        # Mark examples as hard
        if ranks.max() > 0:
            new_hard_qx_list += [qx]
        new_hard_qcid_list = []
        for qx in new_hard_qx_list:
            # New list is in cid format instead of cx format
            # because you should be copying and pasting it
            notes = ' ranks = ' + str(rank_mat[qx])
            qcx = qcx_list[qx]
            qcid = hs.tables.cx2_cid[qcx]
            new_hard_qcid_list += [(qcid, notes)]

    @ArgGaurdFalse
    def print_rowscore():
        print('')
        print('=======================')
        print('[harn] Scores per Query')
        print('=======================')
        for qx in xrange(nQuery):
            bestCFG_X = qx2_argmin_rank[qx]
            min_rank = qx2_min_rank[qx]
            minimizing_cfg_str = indent('\n'.join(cfgx2_lbl[bestCFG_X]), '    ')
            #minimizing_cfg_str = str(bestCFG_X)

            print(qx2_lbl[qx])
            print(' best_rank = %d ' % min_rank)
            if len(cfgx2_lbl) != 1:
                print(' minimizing_cfg_x\'s = %s ' % minimizing_cfg_str)

    print_rowscore()

    #------------

    @ArgGaurdFalse
    def print_hardcase():
        print('===')
        print('--- hard new_hard_qcid_list (w.r.t these configs) ---')
        print('\n'.join(map(repr, new_hard_qcid_list)))
        print('There are %d hard cases ' % len(new_hard_qcid_list))
        print(sorted([x[0] for x in new_hard_qcid_list]))
        print('--- /Scores per Query ---')
    print_hardcase()

    #------------
    # Build Colscore
    X_list = [1, 5]
    # Build a dictionary mapping X (as in #ranks < X) to a list of cfg scores
    nLessX_dict = {int(X): np.zeros(nCfg) for X in iter(X_list)}
    for cfgx in xrange(nCfg):
        ranks = rank_mat[:, cfgx]
        for X in iter(X_list):
            #nLessX_ = sum(np.bitwise_and(ranks < X, ranks >= 0))
            nLessX_ = sum(np.logical_and(ranks < X, ranks >= 0))
            nLessX_dict[int(X)][cfgx] = nLessX_

    @ArgGaurdFalse
    def print_colscore():
        print('')
        print('==================')
        print('[harn] Scores per Config')
        print('==================')
        for cfgx in xrange(nCfg):
            print('[score] %s' % (cfgx2_lbl[cfgx]))
            for X in iter(X_list):
                nLessX_ = nLessX_dict[int(X)][cfgx]
                print('        ' + rankscore_str(X, nLessX_, nQuery))
        print('--- /Scores per Config ---')
    print_colscore()

    #------------

    @ArgGaurdFalse
    def print_latexsum():
        print('')
        print('==========================')
        print('[harn] LaTeX')
        print('==========================')
        # Create configuration latex table
        criteria_lbls = ['#ranks < %d' % X for X in X_list]
        db_name = hs.get_db_name(True)
        cfg_score_title = db_name + ' rank scores'
        cfgscores = np.array([nLessX_dict[int(X)] for X in X_list]).T

        replace_rowlbl = [(' *cfgx *', ' ')]
        tabular_kwargs = dict(title=cfg_score_title, out_of=nQuery,
                              bold_best=True, replace_rowlbl=replace_rowlbl,
                              flip=True)
        tabular_str = latex_formater.make_score_tabular(cfgx2_lbl,
                                                        criteria_lbls,
                                                        cfgscores,
                                                        **tabular_kwargs)
        #latex_formater.render(tabular_str)
        print(tabular_str)
        print('--- /LaTeX ---')
    print_latexsum()

    #------------
    best_rankscore_summary = []
    to_intersect_list = []
    # print each configs scores less than X=thresh
    for X, cfgx2_nLessX in nLessX_dict.iteritems():
        max_LessX = cfgx2_nLessX.max()
        bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
        best_rankscore = '[best_cfg] %d config(s) scored ' % len(bestCFG_X)
        best_rankscore += rankscore_str(X, max_LessX, nQuery)
        best_rankscore_summary += [best_rankscore]
        to_intersect_list += [cfgx2_lbl[bestCFG_X]]

    intersected = to_intersect_list[0] if len(to_intersect_list) > 0 else []
    for ix in xrange(1, len(to_intersect_list)):
        intersected = np.intersect1d(intersected, to_intersect_list[ix])

    @ArgGaurdFalse
    def print_bestcfg():
        print('')
        print('==========================')
        print('[harn] Best Configurations')
        print('==========================')
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in nLessX_dict.iteritems():
            max_LessX = cfgx2_nLessX.max()
            bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
            best_rankscore = '[best_cfg] %d config(s) scored ' % len(bestCFG_X)
            best_rankscore += rankscore_str(X, max_LessX, nQuery)
            best_rankcfg = indent('\n'.join(cfgx2_lbl[bestCFG_X]), '    ')
            print(best_rankscore)
            print(best_rankcfg)

        print('[best_cfg]  %d config(s) are the best of %d total configs' % (len(intersected), nCfg))
        print(indent('\n'.join(intersected), '    '))

        print('--- /Best Configurations ---')
    print_bestcfg()

    #------------

    @ArgGaurdFalse
    def print_rankmat():
        print('')
        print('[harn]-------------')
        print('[harn] labled rank matrix: rows=queries, cols=cfgs:\n%s' % lbld_mat)
        print('[harn]-------------')
    print_rankmat()

    #------------
    print('')
    print('===========================')
    print('[col_score] SUMMARY        ')
    print('===========================')
    print('\n'.join(best_rankscore_summary))
    # Draw results
    rciter = itertools.product(sel_rows, sel_cols)
    for r, c in rciter:
        #print('viewing (r,c)=(%r,%r)' % (r,c))
        res = rc2_res[r, c]
        #res.printme()
        res.show_topN(hs, fnum=fnum)
        fnum += 1
    print('--- /SUMMARY ---')

    print('')
    print('--remember you have --sel_rows and --sel_cols available to you')

#if __name__ == '__main__':
    #import multiprocessing
    #np.set_printoptions(threshold=5000, linewidth=5000, precision=5)
    #multiprocessing.freeze_support()
    #print('[harn]-----------')
    #print('[harn] main()')
    ##main_locals = dev.dev_main()
    #hs = main_locals['hs']
    #qcx_list = main_locals['qcx_list']
    ##test_cfg_name_list = ['vsone_1']
    ##test_cfg_name_list = ['vsmany_3456']
    #test_cfg_name_list = ['vsmany_srule']
    #test_configurations(hs, qcx_list, test_cfg_name_list)
    #exec(df2.present())
