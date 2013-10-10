from __future__ import division, print_function
from os.path import join
import os
import sys
import helpers
import textwrap
import load_data2 as ld2
import match_chips2 as mc2
import report_results2 as rr2
import draw_func2 as df2
import vizualizations as viz
import params
import itertools
import numpy as np
import db_info

def reload_module():
    import imp
    import sys
    print('[exp] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def gen_timsubsets(input_set, M, seed=123456):
    '''generate randomish M-subsets of elements, "far apart"
    Writen by the Tim Peters'''
    import random
    from random import sample
    random.seed(seed)
    elements = sorted(list(input_set))
    allix = set(range(len(elements)))
    takefrom = allix.copy()
    def destructive_sample(n):
        # Remove a random n-subset from takefrom, & return it.
        s = set(sample(takefrom, n))
        takefrom.difference_update(s)
        return s
    while True:
        if len(takefrom) >= M:
            # Get everything from takefrom.
            ix = destructive_sample(M)
        else:
            # We need to take all of takefrom, and more.
            ix = takefrom
            takefrom = allix - ix
            ix |= destructive_sample(M - len(ix))
        assert len(ix) == M
        yield tuple(sorted([elements[i] for i in ix]))

def gen_test_train(input_set, M):
    'Generates randomish unique M-subsets, "far apart"'
    input_set_ = set(input_set)
    seen_subsets = set([])
    timgen = gen_timsubsets(input_set_, M)
    failsafe = 0
    while True: 
        # Generate subset
        test = timgen.next()
        # Check if seen before
        if test in seen_subsets: 
            failsafe += 1
            if failsafe > 100000: 
                raise StopIteration('Generator is meant for M << len(input_set)')
        else: 
            # Mark as seen 
            seen_subsets.add(test)
            failsafe = 0
            train = tuple(sorted(input_set_ - set(test)))
            yield (test, train)

    
def run_experiment(hs=None, free_mem=False, pprefix='[run_expt]', **kwargs):
    'Runs experiment and dumps results. Returns locals={qcx2_res, hs}'
    '''
    import experiments as expt
    from experiments import *
    '''
    print('** Changing print function with pprefix=%r' % (pprefix,))
    def prefix_print(msg):
        helpers.println(helpers.indent(str(msg), pprefix))
    ld2.print = prefix_print
    df2.print = prefix_print
    mc2.print = prefix_print
    rr2.print = prefix_print
    viz.print = prefix_print

    # Load a HotSpotter object with just tables
    if not 'hs' in vars() or hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(ld2.DEFAULT)
        # because we probably plan to at least draw something
        hs.load_chips()
        hs.load_features(load_desc=False)
        hs.set_samples() # default samples
    print('======================')
    print('[expt] Running Experiment on hs:\n'+str(hs.db_name()))
    #print('[expt] Params: \n'+ helpers.indent(params.param_string()))
    print('======================')
    # First load cached results
    qcx2_res, dirty_samp = mc2.load_cached_matches(hs)
    if len(dirty_samp) > 0:
        # Run matching of cached results arent working 
        if hs.matcher is None:
            print('[expt] !! %d dirty queries force the loading of data.' % len(dirty_samp) )
            hs.load_chips()
            hs.load_features(load_desc=True)
            hs.load_matcher()
        # HACK: I need to do this because matcher changes the match_uid
        # This is really bad and needs to be fixed. No changing the damn
        # match_uid!!!
        qcx2_res, dirty_samp = mc2.load_cached_matches(hs)
        qcx2_res = mc2.run_matching(hs, qcx2_res, dirty_samp)
    #if free_mem: 
        # Try to free memory before reporting results
        #hs.free_some_memory()
    allres = rr2.report_all(hs, qcx2_res, **kwargs)
    return locals()

def oxford_philbin07(hs=None):
    # philbin params
    params.__MATCH_TYPE__        = 'bagofwords'
    params.__BOW_NUM_WORDS__     = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    params.__NUM_RERANK__        = [100, 200, 400, 800, 1000][4]
    params.__CHIP_SQRT_AREA__    = None
    params.__XY_THRESH__         = 0.01
    #unsure about checks
    params.BOW_AKMEANS_FLANN_PARAMS = dict(algorithm='kdtree',
                                           trees=8, checks=64) 
    if not 'hs' in vars() or hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(ld2.OXFORD)
        hs.load_chips()
        hs.load_features(load_desc=False)
        hs.set_sample_split_pos(55) # Use the 55 cannonical test cases 
    expt_locals = run_experiment(hs, pprefix='[philbin]', oxford=True)
    return expt_locals

def oxford_bow():
    params.__MATCH_TYPE__     = 'bagofwords'
    params.__CHIP_SQRT_AREA__ = None
    params.__BOW_NUM_WORDS__  = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    db_dir = ld2.OXFORD
    if not 'hs' in vars() or hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(ld2.OXFORD)
        hs.load_chips()
        hs.load_features(load_desc=False)
        hs.set_sample_range(55, None) # Use only database images
    assert min(hs.test_sample_cx) == 55 and max(hs.test_sample_cx) == 5117
    expt_locals = run_experiment(hs, pprefix='[ox-bow]', free_mem=True,
                                 oxford=False, stem=False, matrix=False)
    return expt_locals

def oxford_vsmany():
    params.__MATCH_TYPE__     = 'vsmany'
    params.__CHIP_SQRT_AREA__ = None
    db_dir = ld2.OXFORD
    if not 'hs' in vars() or hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(ld2.OXFORD)
        hs.load_chips()
        hs.load_features(load_desc=False)
        hs.set_sample_range(55, None) # Use only database images
    hs = ld2.HotSpotter(db_dir, samples_range=(55,None))
    expt_locals = run_experiment(hs, pprefix='[ox-vsmany]', free_mem=True,
                                 oxford=False, stem=False, matrix=False)
    return locals()

def far_appart_splits(input_set, M, K):
    split_list = []
    gen = gen_test_train(input_set, M)
    for kx in xrange(K):
        (test, train) = gen.next()
        split_list.append((test,train))
    return split_list


def split_nx2_cxs(test_cxs_list, csplit_size):
    for ix in xrange(len(test_cxs_list)):
        cxs = test_cxs_list[ix]
        num_csplits = len(cxs)//csplit_size
        cxs_splits = far_appart_splits(cxs, csplit_size, num_csplits)
        test_cx_splits.append(cxs_splits)
    max_num_csplits = max(map(len, test_cx_splits))
    # Put them into experiment sets
    jx2_test_cxs = [[] for _ in xrange(max_num_csplits)]
    jx2_index_cxs = [[] for _ in xrange(max_num_csplits)]
    for ix in xrange(len(test_cx_splits)):
        cxs_splits = test_cx_splits[ix]
        for jx in xrange(max_num_csplits):
            if jx >= len(cxs_splits): 
                break
            #ix_test_cxs, ix_index_cxs = cxs_splits[jx]
            ix_index_cxs, ix_test_cxs = cxs_splits[jx]
            jx2_test_cxs[jx].append(ix_test_cxs)
            jx2_index_cxs[jx].append(ix_index_cxs)
    return jx2_test_cxs, jx2_index_cxs
    

def leave_out(expt_func=None, split_test=False, **kwargs):
    '''
    do with TF-IDF on the zebra data set. 
    Let M be the total number of *animals* (not images and not chips) in an experimental data set. 
    Do a series of leave-M-out (M >= 1) experiments on the TF-IDF scoring,
    where the "left out" M are M different zebras, 
    so that there are no images of these zebras in the images used to form the vocabulary.
    The vocabulary is formed from the remaining N-M animals.
    Test how well TF-IDF recognition does with these M animals. 
    Repeat for different subsets of M animals.
    import experiments as expt
    from experiments import *
    '''
    # ---
    # Testing should have animals I have seen and animals I haven't seen. 
    # Make sure num descriptors -per- word is about the same as Oxford 
    # ---
    # Notes from Monday: 
    # 1) Larger training set (see how animals in training do vs animals out of training)
    # 2) More detailed analysis of failures
    # 3) Aggregate scores across different pictures of the same animal
    if not 'expt_func' in vars() or expt_func is None:
        expt_func = run_experiment
    # Load tables
    hs = ld2.HotSpotter(ld2.DEFAULT, load_basic=True)
    # Grab names
    db_names_info = db_info.get_db_names_info(hs)
    nx2_cxs = db_names_info['nx2_cxs']
    valid_nxs = db_names_info['valid_nxs']
    multiton_nxs = db_names_info['multiton_nxs']
    # How to generate samples/splits for names
    num_nsplits = 5
    nsplit_size = (db_names_info['num_names_with_gt']//num_nsplits)
    # How to generate samples/splits for chips
    csplit_size = 1 # number of indexed chips per Jth experiment
    # Generate name splits
    kx2_name_split = far_appart_splits(multiton_nxs, nsplit_size, num_nsplits)
    result_map = {}
    kx = 0
    # run K experiments
    all_cxs = np.hstack(nx2_cxs[list(valid_nxs)])
    for kx in xrange(num_nsplits):
        print('***************')
        print('[expt] Leave M=%r names out iteration: %r/%r' % (nsplit_size, kx+1, num_nsplits))
        print('***************')
        # Get name splits
        (test_nxs, train_nxs) = kx2_name_split[kx]
        # Lock in TRAIN sample
        train_cxs_list = nx2_cxs[list(train_nxs)]
        train_samp = np.hstack(train_cxs_list)
        # Generate chip splits
        test_cxs_list = nx2_cxs[list(test_nxs)]
        test_nChip = map(len, test_cxs_list)
        print('[expt] testnames #cxs stats: %r' % helpers.printable_mystats(test_nChip))
        test_cx_splits  = []
        if not split_test:
            # Chucks version of the test (much simplier and better)
            indx_samp = all_cxs
            train_samp = train_samp
            test_samp = all_cxs
            hs.set_samples(test_samp, train_samp, indx_samp)
            m_label = '[LNO: %r/%r]' % (kx+1, num_nsplits)
            expt_locals = expt_func(hs, pprefix=m_label, **kwargs)
            #result_map[kx] = expt_locals['allres']
    return locals()
'''
        elif split_test:
            jx = 0
            jx2_test_cxs, jx2_index_cxs = split_nx2_cxs(test_cxs_list, csplit_size)
            for jx in xrange(max_num_csplits): # run K*J experiments
                # Lock in TEST and INDEX set
                # INDEX the TRAIN set and a subset of the NOT-TRAIN set
                # TEST chips which have a groundtruth in the INDEX set
                indx_samp = np.hstack(jx2_index_cxs[jx]+[train_samp])
                test_samp = hs.get_valid_cxs_with_name_in_samp(indx_samp)
                hs.set_samples(test_samp, train_samp, indx_samp)
                mj_label = '[LNO:%r/%r;%r/%r]' % (kx+1, num_nsplits, jx+1, max_num_csplits)
                print('[expt] =================')
                print('[expt] M=%r, J=%r' % (nsplit_size,csplit_size))
                expt_locals = expt_func(hs, pprefix=mj_label, **kwargs)
                #result_map[kx] = expt_locals['allres']
                '''
    
def tweak_params(expt_func=None):
    if not 'expt_func' in vars() or expt_func is None:
        expt_func = run_experiment
    xy_thresh_tweaks  = [.05, .01, .005, .001]
    scale_low_tweaks  = [.75, .5, .25]
    scale_high_tweaks = [1.5, 2, 8]
    gen_ = itertools.product(xy_thresh_tweaks, scale_high_tweaks, scale_low_tweaks)
    parameter_list = [tup for tup in gen_]

    total_tests = len(parameter_list)
    result_map = {}
    hs = None
    for count, tup in enumerate(parameter_list):
        print('**********************************************************')
        print('**********************************************************')
        print('========================')
        print('[expt] tweak_params(%d/%d)> param tweak %r ' % (count, total_tests, tup,))
        print('========================')
        rss = helpers.RedirectStdout()
        rss.start()
        xy_thresh, scale_thresh_high, scale_thresh_low = tup
        params.__XY_THRESH__         = xy_thresh
        params.__SCALE_THRESH_LOW__  = scale_thresh_low
        params.__SCALE_THRESH_HIGH__ = scale_thresh_high
        expt_locals = expt_func(hs)
        if 'hs' in expt_locals:
            hs = expt_locals['hs']
        result_map[tup] = expt_locals['allres']
        rss.stop()
    return locals()

def tweak_params_philbin():
    return tweak_params(oxford_philbin07)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('\n\n\n[expt] __main__ = experiments.py')
    print('[expt] sys.argv = %r' % sys.argv)

    # Default to run_experiment
    expt_func = run_experiment

    arg_map = {
        'philbin'        : oxford_philbin07,
        'oxford-bow'     : oxford_bow,
        'oxford-vsmany'  : oxford_vsmany,
        'default'        : run_experiment, 
        'tweak'          : tweak_params,
        'tweak-philbin'  : tweak_params_philbin,
        'leave-out'      : leave_out}

    print ('[expt] Valid experiments are:\n    '+ '\n    '.join(arg_map.keys()))
    argv = sys.argv

    # Change based on user input
    has_arg = False
    for arg in argv:
        if arg in arg_map.keys():
            print('[expt] Running '+str(arg))
            expt_func = arg_map[arg]

    #kwargs = dict(missed_top5=False, rankres=False, stem=False, 
                  #matrix=False, matrix_viz=False, pdf=False, 
                  #hist=False, ttbttf=False, problems=False,
                  #gtmatches=False, oxford=False, no_viz=False):

    kwargs = {}
    if '--noviz' in sys.argv:
        kwargs['no_viz'] = True

    # Do the experiment
    expt_locals = expt_func(**kwargs)
    hs = expt_locals['hs']
    if 'allres' in expt_locals.keys():
        qcx2_res = expt_locals['qcx2_res']
        allres = expt_locals['allres']
        print(allres)

    exec(df2.present())
