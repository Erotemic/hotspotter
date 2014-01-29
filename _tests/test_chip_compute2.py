'''
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import HotSpotterAPI
    import argparse2
    from hsviz import viz
    import chip_compute2 as cc2
    from chip_compute2 import *  # NOQA
    # Debugging vars
    chip_cfg = None
#l')=103.7900s

    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = argparse2.parse_arguments(defaultdb='NAUTS')
    hs = HotSpotterAPI.HotSpotter(args)
    hs.load_tables()
    hs.update_samples()
    # --- LOAD CHIPS --- #
    cc2.load_chips(hs)
    cx = helpers.get_arg('--cx', type_=int)
    if not cx is None:
        tau = np.pi * 2
        hs.change_theta(cx, tau / 8)
        viz.show_chip(hs, cx, draw_kpts=False, fnum=1)
        viz.show_image(hs, hs.cx2_gx(cx), fnum=2)
    else:
        print('usage: feature_compute.py --cx [cx]')
    exec(viz.df2.present())
'''
