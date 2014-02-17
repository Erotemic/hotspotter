if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('=================================')
    print('[viz] __main__ = vizualizations.py')
    print('=================================')
    import main
    hs = main.main()
    cx = helpers.get_arg('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    if cx is not None:
        qcx = cx
    res = hs.query(qcx)
    res.show_top(hs)
    exec(df2.present())
