#---------------
# Display Test Results
#-----------
# Run configuration for each query
def get_test_results(hs, qcxs, qdat, cfgx=0, nCfg=1, nocache_testres=False,
                     test_results_verbosity=2):
    nQuery = len(qcxs)
    dcxs = hs.get_indexed_sample()
    test_uid = qdat.get_query_uid(hs, qcxs)
    cache_dir = join(hs.dirs.cache_dir, 'experiment_harness_results')
    io_kwargs = dict(dpath=cache_dir, fname='test_results', uid=test_uid,
                     ext='.cPkl')

    if test_results_verbosity == 2:
        print('[harn] test_uid = %r' % test_uid)

    # High level caching
    if not params.args.nocache_query and (not nocache_testres):
        qx2_bestranks = io.smart_load(**io_kwargs)
        if qx2_bestranks is None:
            print('[harn] qx2_bestranks cache returned None!')
        elif len(qx2_bestranks) != len(qcxs):
            print('[harn] Re-Caching qx2_bestranks')
        elif not qx2_bestranks is None:
            return qx2_bestranks, [[{0: None}]] * nQuery
        #raise Exception('cannot be here')
    mc3.ensure_nn_index(hs, qdat, dcxs)
    nPrevQ = nQuery * cfgx
    qx2_bestranks = []

    # Make progress message
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST %d/%d
    ---------------------''')
    mark_progress = util.simple_progres_func(test_results_verbosity, msg, '.')
    total = nQuery * nCfg
    # Perform queries
    TEST_INFO = True
    # Query Chip / Row Loop
    for qx, qcx in enumerate(qcxs):
        count = qx + nPrevQ + 1
        mark_progress(count, total)
        if TEST_INFO:
            print('qcx=%r. quid=%r' % (qcx, qdat.get_uid()))
        try:
            res_list = mc3.execute_query_safe(hs, qdat, [qcx], dcxs)
        except mf.QueryException as ex:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Harness caught Query Exception: ')
            print(ex)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if params.args.strict:
                raise
            else:
                qx2_bestranks += [[-1]]
                continue

        assert len(res_list) == 1
        bestranks = []
        for qcx2_res in res_list:
            assert len(qcx2_res) == 1
            res = qcx2_res[qcx]
            gt_ranks = res.get_gt_ranks(hs=hs)
            #print('[harn] cx_ranks(/%4r) = %r' % (nChips, gt_ranks))
            #print('[harn] cx_ranks(/%4r) = %r' % (NMultiNames, gt_ranks))
            #print('ns_ranks(/%4r) = %r' % (nNames, gt_ranks))
            _bestrank = -1 if len(gt_ranks) == 0 else min(gt_ranks)
            bestranks += [_bestrank]
        # record metadata
        qx2_bestranks += [bestranks]
        if qcx % 4 == 0:
            sys.stdout.flush()
    print('')
    qx2_bestranks = np.array(qx2_bestranks)
    # High level caching
    util.ensuredir(cache_dir)
    io.smart_save(qx2_bestranks, **io_kwargs)

    return qx2_bestranks

