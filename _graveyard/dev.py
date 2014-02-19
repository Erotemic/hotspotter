from __init__ import *
#rrr()

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


def history_entry(database='', cid=-1, ocids=[], notes='', cx=-1):
    return (database, cid, ocids, notes)

# A list of poster child examples. (curious query cases)
GZ_greater1_cid_list = [140, 297, 306, 311, 425, 441, 443, 444, 445, 450, 451,
                        453, 454, 456, 460, 463, 465, 501, 534, 550, 662, 786,
                        802, 838, 941, 981, 1043, 1046, 1047]
HISTORY = [
    history_entry('TOADS', cx=32),
    history_entry('NAUTS', 1,    [],               notes='simple eg'),
    history_entry('WDOGS', 1,    [],               notes='simple eg'),
    history_entry('MOTHERS', 69, [68],             notes='textured foal (lots of bad matches)'),
    history_entry('MOTHERS', 28, [27],             notes='viewpoint foal'),
    history_entry('MOTHERS', 53, [54],             notes='image quality'),
    history_entry('MOTHERS', 51, [50],             notes='dark lighting'),
    history_entry('MOTHERS', 44, [43, 45],         notes='viewpoint'),
    history_entry('MOTHERS', 66, [63, 62, 64, 65], notes='occluded foal'),
]

#MANUAL_GZ_HISTORY = [
    #history_entry('GZ', 662,     [262],            notes='viewpoint / shadow (circle)'),
    #history_entry('GZ', 1046,    [],               notes='extreme viewpoint #gt=2'),
    #history_entry('GZ', 838,     [801, 980],       notes='viewpoint / quality'),
    #history_entry('GZ', 501,     [140],            notes='dark lighting'),
    #history_entry('GZ', 981,     [802],            notes='foal extreme viewpoint'),
    #history_entry('GZ', 306,     [112],            notes='occlusion'),
    #history_entry('GZ', 941,     [900],            notes='viewpoint / quality'),
    #history_entry('GZ', 311,     [289],            notes='quality'),
    #history_entry('GZ', 1047,    [],               notes='extreme viewpoint #gt=4'),
    #history_entry('GZ', 297,     [301],            notes='quality'),
    #history_entry('GZ', 786,     [787],            notes='foal #gt=11'),
    #history_entry('GZ', 534,     [411, 727],       notes='LNBNN failure'),
    #history_entry('GZ', 463,     [173],            notes='LNBNN failure'),
    #history_entry('GZ', 460,     [613, 460],       notes='background match'),
    #history_entry('GZ', 465,     [589, 460],       notes='background match'),
    #history_entry('GZ', 454,     [198, 447],       notes='forground match'),
    #history_entry('GZ', 445,     [702, 435],       notes='forground match'),
    #history_entry('GZ', 453,     [682, 453],       notes='forground match'),
    #history_entry('GZ', 550,     [551, 452],       notes='forground match'),
    #history_entry('GZ', 450,     [614],            notes='other zebra match'),
#]

                                                                                ##csum, pl, plw, borda
#AUTO_GZ_HISTORY = map(lambda tup: tuple(['GZ'] + list(tup)), [
    #(662,   [263],                              'viewpoint / shadow (circle) ranks = [16 20 20 20]'),
    #(1046,  [],                                 'extreme viewpoint #gt=2 ranks     = [592 592 592 592]'),
    #(838,   [802, 981],                         'viewpoint / quality ranks         = [607 607 607 607]'),
    #(501,   [141],                              'dark lighting ranks               = [483 483 483 483]'),
    #(802,   [981],                              'viewpoint / quality /no matches   = [722 722 722 722]'),
    #(907,   [828, 961],                         'occluded but (spatial verif)      = [645 645 645 645]'),
    #(1047,  [],                                 'extreme viewpoint #gt=4 ranks     = [582 582 582 582]'),
    #(16,    [635],                              'NA ranks                          = [839 839 839 839]'),
    #(140,   [501],                              'NA ranks                          = [194 194 194 194]'),
    #(981,   [803],                              'foal extreme viewpoint ranks      = [ 8  9  9 11]'),
    #(425,   [662],                              'NA ranks                          = [21 33 30 34]'),
    #(681,   [198, 454, 765],                    'NA ranks                          = [2 6 6 6]'),
    #(463,   [174],                              'LNBNN failure ranks               = [3 0 3 0]'),
    #(306,   [113],                              'occlusion ranks                   = [1 1 1 1]'),
    #(311,   [290],                              'quality ranks                     = [1 2 1 2]'),
    #(460,   [614, 461],                         'background match ranks            = [2 1 2 1]'),
    #(465,   [590, 461],                         'background match ranks            = [3 0 3 0]'),
    #(454,   [199, 448],                         'forground match ranks             = [5 3 3 2]'),
    #(445,   [703, 436],                         'forground match ranks             = [1 2 2 2]'),
    #(453,   [683, 454],                         'forground match ranks             = [2 3 4 0]'),
    #(550,   [552, 453],                         'forground match ranks             = [5 5 5 4]'),
    #(450,   [615],                              'other zebra match ranks           = [3 4 4 4]'),
    #(95,    [255],                              'NA ranks                          = [2 5 5 5]'),
    #(112,   [306],                              'NA ranks                          = [1 2 2 2]'),
    #(183,   [178],                              'NA ranks                          = [1 2 2 2]'),
    #(184,   [34, 39, 227, 619],                 'NA ranks                          = [1 1 1 1]'),
    #(253,   [343],                              'NA ranks                          = [1 1 1 1]'),
    #(276,   [45, 48],                           'NA ranks                          = [1 0 1 0]'),
    #(277,   [113, 124],                         'NA ranks                          = [1 0 1 0]'),
    #(289,   [311],                              'NA ranks                          = [2 1 2 1]'),
    #(339,   [315],                              'NA ranks                          = [1 1 1 1]'),
    #(340,   [317],                              'NA ranks                          = [1 0 1 0]'),
    #(415,   [408],                              'NA ranks                          = [1 3 2 4]'),
    #(430,   [675],                              'NA ranks                          = [1 0 1 0]'),
    #(436,   [60, 61, 548, 708, 760],            'NA ranks                          = [1 0 0 0]'),
    #(441,   [421],                              'NA ranks                          = [5 5 6 5]'),
    #(442,   [693, 777],                         'NA ranks                          = [1 0 1 0]'),
    #(443,   [420, 478],                         'NA ranks                          = [5 4 6 4]'),
    #(444,   [573],                              'NA ranks                          = [5 3 5 3]'),
    #(446,   [565, 678, 705],                    'NA ranks                          = [1 0 0 0]'),
    #(451,   [541, 549],                         'NA ranks                          = [2 0 1 0]'),
    #(456,   [172, 174, 219, 637],               'NA ranks                          = [3 1 2 0]'),
    #(661,   [59],                               'NA ranks                          = [0 4 4 4]'),
    #(720,   [556, 714],                         'NA ranks                          = [1 0 0 0]'),
    #(763,   [632],                              'NA ranks                          = [0 6 0 6]'),
    #(1044,  [845, 878, 927, 1024, 1025, 1042],  'NA ranks                          = [1 0 0 0]'),
    #(1045,  [846, 876],                         'NA ranks                          = [1 0 1 0]'),
#])
#HISTORY += AUTO_GZ_HISTORY


#def mothers_problem_pairs():
    #'''MOTHERS Dataset: difficult (qcx, cx) query/result pairs'''
    #viewpoint = [( 16, 17), (19, 20), (73, 71), (75, 78), (108, 112), (110, 108)]
    #quality = [(27, 26),  (52, 53), (67, 68), (73, 71), ]
    #lighting = [(105, 104), ( 49,  50), ( 93,  94), ]
    #confused = []
    #occluded = [(64, 65), ]
    #return locals()


