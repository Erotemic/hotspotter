'''
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[fc2] __main__ = feature_compute2.py')
    import main
    import HotSpotterAPI
    from hsviz import viz
    import feature_compute2 as fc2
    from feature_compute2 import *  # NOQA
    # Debugging vars
    feat_cfg = None
    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = main.parse_arguments(db='NAUTS')
    hs = HotSpotterAPI.HotSpotter(args)
    hs.load_tables()
    hs.set_samples()
    # --- LOAD CHIPS --- #
    hs.load_configs()
    hs.load_chips()
    # --- LOAD FEATURES --- #
    load_features(hs)
    cx = helpers.get_arg('--cx', type_=int)
    delete_features = '--delete-features' in sys.argv
    nRandKpts = helpers.get_arg('--nRandKpts', type_=int)
    if delete_features:
        fc2.clear_feature_cache(hs)
    if not cx is None:
        viz.show_chip(hs, cx, nRandKpts=nRandKpts)
    else:
        print('usage: feature_compute.py --cx [cx] --nRandKpts [num] [--delete-features]')

    exec(viz.present())
'''
