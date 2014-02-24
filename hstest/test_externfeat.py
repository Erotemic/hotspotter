'''
from os.path import expanduser, exists, split
if __name__ == '__main__':
    print('[TPL] Test Extern Features')
    import multiprocessing
    multiprocessing.freeze_support()

    def ensure_hotspotter():
        import matplotlib
        matplotlib.use('Qt4Agg', warn=True, force=True)
        # Look for hotspotter in ~/code
        hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
        if not exists(hotspotter_dir):
            print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
        # Append hotspotter location (not dir) to PYTHON_PATH (i.e. sys.path)
        hotspotter_location = split(hotspotter_dir)[0]
        sys.path.append(hotspotter_location)

    # Import hotspotter io and drawing
    ensure_hotspotter()
    from hsviz import draw_func2 as df2
    #from hsviz import viz
    from hscom import fileio as io

    # Read Image
    img_fpath = realpath('lena.png')
    image = io.imread(img_fpath)

    def spaced_elements(list_, n):
        indexes = np.arange(len(list_))
        stride = len(indexes) // n
        return list_[indexes[0:-1:stride]]

    def test_detect(n=None, fnum=1, old=True):
        from hsviz import interact
        try:
            # Select kpts
            detect_kpts_func = detect_kpts_old if old else detect_kpts_new
            kpts, desc = detect_kpts_func(img_fpath, {})
            kpts_ = kpts if n is None else spaced_elements(kpts, n)
            desc_ = desc if n is None else spaced_elements(desc, n)
            # Print info
            np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
            print('----')
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            print(kpts_)
            print('----')
            # Draw kpts
            interaction.interact_keypoints(image, kpts_, desc_, fnum)
            #df2.imshow(image, fnum=fnum)
            #df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                           #ell_color='distinct', arrow=True, rect=True)
            df2.set_figtitle('old' if old else 'new')
        except Exception as ex:
            import traceback
            traceback.format_exc()
            print(ex)
        return locals()

    test_detect(n=10, fnum=1, old=True)
    test_detect(n=10, fnum=2, old=False)
    exec(df2.present())
'''
