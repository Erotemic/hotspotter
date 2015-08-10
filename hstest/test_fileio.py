
# --- Main Test ---

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    """
    data1 = (255 * np.random.rand(10000, 10000)).astype(np.uint8)
    data2 = np.random.rand(10000, 10000).astype(np.float64)
    data3 = (255 * np.random.rand(10000, 10000)).astype(np.int32)

    print('[io] Created arrays')
    save_npy.ext = '.npy'
    save_npz.ext = '.npz'
    save_cPkl.ext = '.cPkl'
    save_pkl.ext = '.pkl'

    load_npy.ext = '.npy'
    load_npz.ext = '.npz'
    load_cPkl.ext = '.cPkl'
    load_pkl.ext = '.pkl'

    fpath_list = ['data1', 'data2', 'data3']
    data_list  =  [data1, data2, data3]

    save_func_list = [save_npy, save_npz, save_cPkl, save_pkl]
    load_func_list = [load_npy, load_npz, load_cPkl, load_pkl]

    fpath = '/media/Store/data/work/Oxford_Buildings/.hs_internals/'

    # Test Save
    for save_func in save_func_list:
        print('[io] Testing: ' + save_func.__name__)
        print('[io]  withext: ' + save_func.ext)
        tt_total = util.tic(save_func.__name__)

        for fpath, data, in zip(fpath_list, data_list):
            fpath += save_func.ext
            tt_single = util.tic(fpath)
            save_func(fpath, data)
            util.toc(tt_single)
        util.toc(tt_total)
        print('------------------')

    # Test memory:
    for save_func in save_func_list:
        for fpath in fpath_list:
            fpath += save_func.ext
            print(util.file_megabytes_str(fpath))

    # Test Load
    for load_func in load_func_list:
        print('Testing: ' + load_func.__name__)
        print(' withext: ' + load_func.ext)
        tt_total = util.tic(load_func.__name__)

        for fpath, data, in zip(fpath_list, data_list):
            fpath += load_func.ext
            tt = util.tic(fpath)
            data2 = load_func(fpath)
            util.toc(tt)
        util.toc(tt_total)
        print('------------------')
    print(util.file_megabytes_str(fpath))

    tic = util.tic
    toc = util.toc

    #tt = tic(fpath_py)
    #with open(fpath, 'wb') as file_:
        #npz = np.load(file_, fpath)
        #data = npz['arr_0']
        #npz.close()
    #toc(tt)

    tt = tic(fpath_py)
    with open(fpath_py, 'wb') as file_:
        np.save(file_, data)
    toc(tt)

    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'wb') as file_:
        np.savez(file_, data)
    toc(tt)

    tt = tic(fpath_py)
    with open(fpath_py, 'rb') as file_:
        npy_data = np.load(file_)
    toc(tt)
    print(util.file_megabytes_str(fpath_py))


    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'rb') as file_:
        npz = np.load(file_)
        npz_data = npz['arr_0']
        npz.close()
    toc(tt)
    print(util.file_megabytes_str(fpath_pyz))

    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'rb') as file_:
        npz = np.load(file_, mmap_mode='r+')
        npz_data = npz['arr_0']
        npz.close()
    toc(tt)

    tt = util.tic(fpath)
    data2 = load_func(fpath)

    with Timer():
        with open(fpath, 'rb') as file_:
            npz = np.load(file_, mmap_mode='r')
            data = npz['arr_0']
            npz.close()

    with Timer():
        with open(fpath, 'rb') as file_:
            npz2 = np.load(file_, mmap_mode=None)
            data2 = npz['arr_0']
            npz2.close()
    """
