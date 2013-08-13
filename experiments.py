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

helpers.__PRINT_CHECKS__ = True

# TODO: No resize chips
# TODO: Orientation assignment / Mikj detectors
# I guess no orientation
#These three models take advantage of the fact that images are usually
#displayed on the web with the correct (upright) orientation. For this
#reason, we have not allowed for in-plane image rotations.
def philbin07_oxford():
    # The vocab sizes run by philbin et al 
    vocab_sizes = [1e4, 2e4, 5e4, 1e6, 1.25e6] 
    rerank_nums = [100, 200, 400, 800]
    # They use 8 trees in for their AKMEANS. Unsure how many checks
    philbin_akmeans_params = {'algorithm':'kdtree',
                              'trees'    :8,
                              'checks'   :64}
    # Degrees of freedom is fixed at 5 for now
    # dof = [3, 4, 5]
    params.__MATCH_TYPE__        = 'bagofwords'
    params.__BOW_NUM_WORDS__     = vocab_sizes[3]
    params.__NUM_RERANK__        = rerank_nums[3]
    params.__CHIP_SQRT_AREA__    = None
    params.__BOW_AKMEANS_FLANN_PARAMS__ = philbin_akmeans_params  # For AKMEANS
    # Load Oxford Dataset
    dbdir = load_data2.OXFORD
    hs = load_data2.HotSpotter(dbdir, load_matcher=False)
    # Load train / test / database samples
    hs.load_test_train_database_samples_from_file()
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
    report_results2.dump_qcx_tt_bt_tf(hs, qcx2_res)

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
    print params.param_string()
    db_dir = load_data2.DEFAULT
    print(textwrap.dedent('''
    ======================
    Running Experiment on: %r
    ======================''' % db_dir))
    algo_uid = params.get_algo_uid()
    hs = load_data2.HotSpotter(db_dir)
    qcx2_res = mc2.run_matching(hs)
    report_results2.write_rank_results(hs, qcx2_res, SV=True)
    report_results2.write_rank_results(hs, qcx2_res, SV=False)
    report_results2.dump_qcx_tt_bt_tf(hs, qcx2_res)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import load_data2
    freeze_support()

    arg_map = {
        'philbin07' : philbin07_oxford,
        'test'  : run_experiment }

    print ('Valid arguments are:\n    '+ '\n    '.join(arg_map.keys()))
    for argv in sys.argv:
        if argv in arg_map.keys():
            print('Running '+str(argv))
            arg_map[argv]()
