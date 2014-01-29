

def test():
    import dev
    from hsviz import viz
    main_locals = dev.dev_main()
    hs  = main_locals['hs']        # hotspotter api
    qcx = main_locals['qcx']       # query chip index
    viz.viz_spatial_verification(hs, qcx)


#if __name__ == '__main__':
    #import multiprocessing
    #multiprocessing.freeze_support()
    #import matplotlib
    #matplotlib.use('Qt4Agg')
    #from hsviz import draw_func2 as df2
    #print('[sc2] __main__ = spatial_verification2.py')
    #test()
    #exec(df2.present(num_rc=(1, 1), wh=2500))
