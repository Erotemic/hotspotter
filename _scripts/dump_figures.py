from __future__ import print_function, division
from dbgimport import *


species_map = {
    'NAUT_Dan':                 'Nautiluses',
    'Elephants_Stewart':        'Elephants',
    'JAG_Kelly':                'Jaguars',
    'LF_all':                   'Lionfishs',
    'WD_Siva':                  'Wild Dogs',
    'SL_Siva':                  'Snow Leopards',
    'Seals':                    'Seals',
    'Wildebeast':               'Wildebeast',
    'Frogs':                    'Frogs',
    'WY_Toads':                 'Wyoming Toads',
    'WS_hard':                  'Whale Sharks',
    'HSDB_zebra_with_mothers':  'Plains Zebras',
    'GZ_Dan_Group1':            'Grevys Zebras',
    'RhinoTest':                'Rhinos',
    'Amur':                     'Amur Leopards',
    'GIR_Tanya':                'Giraffes',
}

positive_examples = {
    'NAUT_Dan':                 [(3, [6])],
    'Elephants_Stewart':        [(34, [32, 29]), (135, [134]), (86, [84]), (73, [99])],
    'JAG_Kelly':                [(21, [24])],
    'LF_all':                   [(769, [770])],
    'WD_Siva':                  [(92, [93]), (39, [42]), (179, [178])],
    'SL_Siva':                  [(44, [39]), ],
    'Seals':                    [(19, [22]), ],
    'Wildebeast':               [(20, [19]), ],
    'Frogs':                    [(2, [3])],
    'WY_Toads':                 [(802, [803, 801]), ],
    'WS_hard':                  [(20, [19]), ],
    'HSDB_zebra_with_mothers':  [(1, [2, 3, 4]), ],
    'GZ_Dan_Group1':            [
        (74, [77]),
        (36, [39]),
        (51, [61]),
        (47, [71]),
        (19, [16]),
        #
        (44,  [115]),
        (115, [71]),
        (71,  [47]),
        (47,  [81]),
        (81,  [58]),
        (58,  [56]),
        (56,  [51]),
        (51,  [78]),
    ],
    'RhinoTest':                [(6, [2])],
    'Amur':                     [(28, [25])],
    'GIR_Tanya':                [(16, [17])],
}


with_histeq = ['Frogs', 'Elephants_Stewart']

dbname = 'NAUT_Dan'
cid_pair = (3, 6)

rootoutput_dir = expanduser('~/Dropbox/Latex/proposal/auto_figures2')

DUMP_NAMES = True
DUMP_CHIPRES_IMAGEVIEW = True
DUMP_CHIPRES_CHIPVIEW = True
DUMP_QUERY_RESULTS = True


# only dump certain datasets
DATASETS_LIST = species_map.keys()
#USE_DATASETS = ['GZ_Dan_Group1']
positive_examples = {key: positive_examples[key] for key in DATASETS_LIST}


def dump_fig(output_dir):
    util.ensuredir(output_dir)
    df2.adjust_subplots_safe()
    df2.save_figure(fpath=output_dir, usetitle=True)
    df2.reset()


for dbname, query_list in positive_examples.iteritems():
    import argparse2
    # FIX ARGPARSE CRAP
    args = argparse2.parse_arguments()
    db_dir = join(expanduser('~/data/work'), dbname)
    args.dbdir = db_dir
    hs = api.HotSpotter(args)
    if dbname in with_histeq:
        hs.prefs.chip_cfg.histeq = True
        hs.prefs.chip_cfg.adapteq = True
    else:
        hs.prefs.chip_cfg.histeq = False
        hs.prefs.chip_cfg.adapteq = False
    hs.load()

    shown_names = []

    for cid_pair in query_list:

        qcid, cid_list = cid_pair
        qcx = hs.cid2_cx(qcid)
        print('\n   --- SHOW %s QCID=%r --- \n' % (dbname, qcid))

        output_dir = join(rootoutput_dir, dbname)

        res = hs.query(qcx)

        if DUMP_QUERY_RESULTS:
            df2.FIGSIZE = df2.FIGSIZE_MED
            res.show_top(hs)
            dump_fig(output_dir)

        nx = hs.tables.cx2_nx[qcx]

        for cid in cid_list:
            print('\n   --- SHOW %s QCID=%r vs CID=%r --- \n' % (dbname, qcid, cid))
            cx = hs.cid2_cx(cid)
            df2.FIGSIZE = df2.FIGSIZE_SQUARE

            if DUMP_CHIPRES_IMAGEVIEW:
                res.show_chipres(hs, cx, in_image=True)
                df2.set_figtitle('Query Result nx=%r %s_IMAGE' % (nx, hs.vs_str(qcx, cx)))
                dump_fig(output_dir)

            if DUMP_CHIPRES_CHIPVIEW:
                res.show_chipres(hs, cx, in_image=False)
                df2.set_figtitle('Query Result nx=%r %s_CHIP' % (nx, hs.vs_str(qcx, cx)))
                dump_fig(output_dir)

        if not nx in shown_names and DUMP_NAMES:
            df2.FIGSIZE = df2.FIGSIZE_SQUARE
            print('\n   --- SHOW %s NX=%r --- \n' % (dbname, nx))
            shown_names.append(nx)
            viz.show_name(hs, nx)
            dump_fig(output_dir)

#util.vd(rootoutput_dir)
