# We have 5,202,157 descriptors
# They claim to have 16,334,970 descriptors
from os.path import join
import helpers
import textwrap
import load_data2
import match_chips2 as mc2
import numpy as np
import os
import params
import report_results2
import sys 

#helpers.__PRINT_CHECKS__ = True

# TODO: No resize chips
# TODO: Orientation assignment / Mikj detectors
# I guess no orientation
#These three models take advantage of the fact that images are usually
#displayed on the web with the correct (upright) orientation. For this
#reason, we have not allowed for in-plane image rotations.
def philbin07_oxford():
    params.__MATCH_TYPE__        = 'bagofwords'
    params.__BOW_NUM_WORDS__     = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    params.__NUM_RERANK__        = [100, 200, 400, 800, 1000][3]
    params.__CHIP_SQRT_AREA__    = None
    params.__BOW_AKMEANS_FLANN_PARAMS__ = dict(algorithm='kdtree',
                                               trees=8, checks=64)
    # I'm not sure if checks parameter is set correctly
    dbdir = load_data2.OXFORD
    hs = load_data2.HotSpotter(dbdir, load_matcher=False)
    # Use the 55 cannonical test cases 
    hs.load_test_train_database_samples_from_file(test_sample_fname='test_sample55.txt')
    # Quick Sanity Checks
    db_sample_cx = hs.database_sample_cx
    tr_sample_cx = hs.train_sample_cx
    te_sample_cx = hs.test_sample_cx
    assert db_sample_cx == tr_sample_cx
    assert len(set(te_sample_cx)) == 55
    print('Database shape: '+str(np.vstack(hs.feats.cx2_desc[db_sample_cx]).shape))
    # Load / Build Vocabulary
    hs.load_matcher()
    # Run the matching
    qcx2_res = mc2.run_matching(hs)
    report_results2.write_oxsty_mAP_results(hs, qcx2_res, SV=True)
    report_results2.write_oxsty_mAP_results(hs, qcx2_res, SV=False)
    report_results2.write_rank_results(hs, qcx2_res, SV=True)
    report_results2.write_rank_results(hs, qcx2_res, SV=False)
    report_results2.dump_all(hs, qcx2_res)
    

db_dir = load_data2.JAGUARS
def run_experiment():
    from os.path import join
    import helpers
    import textwrap
    import load_data2
    import match_chips2 as mc2
    import numpy as np
    import os
    import params
    import report_results2
    import sys 
    db_dir = load_data2.DEFAULT
    print(textwrap.dedent('''
    ======================
    Running Experiment on: %r
    ======================''' % db_dir))
    print params.param_string()

    hs = load_data2.HotSpotter(db_dir)
    qcx2_res = mc2.run_matching(hs)
    report_results2.write_rank_results(hs, qcx2_res, SV=True)
    report_results2.write_rank_results(hs, qcx2_res, SV=False)
    #report_results2.dump_qcx_tt_bt_tf(hs, qcx2_res)
    report_results2.plot_summary_visualizations(hs, qcx2_res)
    report_results2.dump_all(hs, qcx2_res)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import load_data2
    freeze_support()

    arg_map = {
        'philbin' : philbin07_oxford,
        'default'   : run_experiment }

    print ('Valid arguments are:\n    '+ '\n    '.join(arg_map.keys()))

    has_arg = False
    for argv in sys.argv:
        if argv in arg_map.keys():
            print('Running '+str(argv))
            arg_map[argv]()
            has_arg = True

    if not has_arg:
        run_experiment()
