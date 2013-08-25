from __init__ import *
rrr()

def reload_module():
    import imp
    import sys
    print('reloading dev: '+__name__)
    imp.reload(sys.modules[__name__])

def dev():
    execstr = open("dev.py").read()
    #exec(execstr)
    return execstr

def test_data():
    db_dir = ld2.DEFAULT
    print('\n\n * ================== * ')
    print('Loading test data: '+db_dir)
    #from __init__ import *
    #rsout = helpers.RedirectStdout()
    #rsout.start()
    #with helpers.RedirectStdout('test_data>'):
    hs = ld2.HotSpotter(db_dir)
    #record = rsout.stop()
    #print helpers.indent(record, 'test_data>')
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    # local noramlize debugging
    chip = hs.get_chip_pil(qcx)
    print('* ================== *\n\n')
    return locals()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    #helpers.explore_stack()
    print(textwrap.dedent('''
    __+----------------------+__
      +~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ +
      !--- * DEV SCRIPT * ---!
      + ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~+
    __|______________________|__
                        '''))
    test_locals = search_stack_for_localvar('test_locals')
    if test_locals is None: 
        test_locals = test_data()

    test_locals_keys = test_locals.keys()
    # Format locals by type and name
    torepl = ["<type '", "'>", "<class '"]
    def val2typestr(val): return helpers.list_replace(str(type(val)), torepl)
    type_val_iter = ('\n    %8s #type=%s;' % (key, val2typestr(val)) 
            for key, val in test_locals.iteritems())
    test_locals_info = ''.join(type_val_iter)
    print('===============')
    print('Adding test locals to namespace: '+test_locals_info)
    dict_ = test_locals
    exec(helpers.execstr_dict(dict_, 'test_locals'))

    experiments.param_config1()
    print('\n\n===============')
    print('Parameters:')
    print params.param_string()

    func = experiments.run_experiment
    run_expt_str = helpers.execstr_func(func)
    print('Try these: ')
    print('exec(run_expt_str)')

    __QTCONSOLE__ = sys.platform == 'win32'
    if not __QTCONSOLE__:
        exec(helpers.IPYTHON_EMBED_STR)

