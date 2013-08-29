from os.path import join
import helpers
import textwrap
import load_data2 as ld2
import match_chips2 as mc2
import numpy as np
import os
import params
import report_results2 as rr2
import sys 
import draw_func as df2

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

def param_config1():
    params.__RANK_EQ__ = True

def param_config2():
    params.__RANK_EQ__ = False

def run_experiment(hs=None):
    '''
    Runs experiment and report result
    returns qcx2_res, hs
    '''
    db_dir = ld2.DEFAULT
    if not hs is None:
        db_dir = hs.db_dir
    print(textwrap.dedent('''
    ======================
    expts> Running Experiment on: %r
    Params: %s
    ======================''' % (db_dir,helpers.indent(params.param_string()))))
    hs = hs if not hs is None else ld2.HotSpotter(db_dir)
    qcx2_res = mc2.run_matching(hs)
    allres = rr2.report_all(hs, qcx2_res)
    return locals()

def oxford_philbin07():
    params.__MATCH_TYPE__        = 'bagofwords'
    params.__BOW_NUM_WORDS__     = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    params.__NUM_RERANK__        = [100, 200, 400, 800, 1000][3]
    params.__CHIP_SQRT_AREA__    = None
    params.BOW_AKMEANS_FLANN_PARAMS = dict(algorithm='kdtree',
                                               trees=8, checks=64)
    # I'm not sure if checks parameter is set correctly
    db_dir = ld2.OXFORD
    hs = ld2.HotSpotter(db_dir, load_matcher=False)
    # Use the 55 cannonical test cases 
    hs.load_file_samples(test_sample_fname='test_sample55.txt')
    # Quick Sanity Checks
    db_sample_cx = hs.database_sample_cx
    tr_sample_cx = hs.train_sample_cx
    te_sample_cx = hs.test_sample_cx
    assert db_sample_cx == tr_sample_cx
    assert len(set(te_sample_cx)) == 55
    print('expts> Database shape: '+str(np.vstack(hs.feats.cx2_desc[db_sample_cx]).shape))
    # Load / Build Vocabulary
    hs.load_matcher()
    # Run the matching
    qcx2_res = mc2.run_matching(hs)
    allres = rr2.report_all(hs, qcx2_res, oxford=True)
    return locals()

def oxford_bow():
    params.__MATCH_TYPE__     = 'bagofwords'
    params.__CHIP_SQRT_AREA__ = None
    params.__BOW_NUM_WORDS__  = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    db_dir = ld2.OXFORD
    hs = ld2.HotSpotter()
    hs.load_tables(db_dir)
    hs.load_chips()
    hs.load_features(load_desc=False)
    hs.load_file_samples()
    assert min(hs.test_sample_cx) == 55
    assert max(hs.test_sample_cx) == 5117
    qcx2_res = mc2.run_matching(hs)
    hs.free_some_memory()
    allres = rr2.report_all(hs, qcx2_res, oxford=False, stem=False, matrix=False)
    return locals()

def oxford_vsmany():
    params.__MATCH_TYPE__     = 'vsmany'
    params.__CHIP_SQRT_AREA__ = None
    db_dir = ld2.OXFORD
    hs = ld2.HotSpotter(db_dir, samples_from_file=True)
    qcx2_res = mc2.run_matching(hs)
    allres = rr2.report_all(hs, qcx2_res, oxford=False, stem=False, matrix=False)
    return locals()
    
def mothers_vsmany():
    params.__MATCH_TYPE__     = 'vsmany'
    return run_experiment()

def mothers_bow():
    params.__MATCH_TYPE__     = 'bagofwords'
    return run_experiment()

def demo():
    pass
#ld2.DEFAULT

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    arg_map = {
        'philbin'       : oxford_philbin07,
        'oxford-bow'    : oxford_bow,
        'oxford-vsmany' : oxford_vsmany,
        'mothers-bow'    : mothers_bow,
        'mothers-vsmany' : mothers_vsmany,
        'default'       : run_experiment }

    print ('expts> Valid arguments are:\n    '+ '\n    '.join(arg_map.keys()))

    has_arg = False
    for argv in sys.argv:
        if argv in arg_map.keys():
            print('expts> Running '+str(argv))
            expt_func = arg_map[argv]
            expt_locals = expt_func()
            hs = expt_locals['hs']
            qcx2_res = expt_locals['qcx2_res']
            allres = expt_locals['allres']
            has_arg = True
        #elif argv.find('param') > -1:
            #param_config1()
            #expt_locals = run_experiment()

    if not has_arg:
        expt_locals2 = run_experiment()


    exec(df2.present())
