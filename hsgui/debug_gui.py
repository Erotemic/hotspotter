from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[back]')


def reload_all_modules():
    # TODO Integrate this better
    print('===========================')
    print('[dev] performing dev_reload')
    print('---------------------------')
    from hotspotter import DataStructures as ds
    from hotspotter import algos
    from hotspotter import load_data2 as ld2
    from hotspotter import chip_compute2 as cc2
    from hotspotter import feature_compute2 as fc2
    from hotspotter import match_chips3 as mc3
    from hotspotter import matching_functions as mf
    from hotspotter import nn_filters
    from hotspotter import report_results2 as rr2
    from hotspotter import voting_rules2 as vr2
    # Common
    from hscom import fileio as io  # NOQA
    from hscom import helpers  # NOQA
    from hscom import cross_platform as cplat
    # Viz
    from hsviz import draw_func2 as df2  # NOQA
    from hsviz import interact  # NOQA
    from hsviz import viz
    # GUI
    from hsgui import guitools  # NOQA
    from hsgui import guifront  # NOQA
    from hsgui import guiback  # NOQA
    # Self
    rrr()
    # com
    helpers.rrr()
    io.rrr()
    cplat.rrr()
    # hotspotter
    ld2.rrr()
    ds.rrr()
    mf.rrr()
    nn_filters.rrr()
    mc3.rrr()
    vr2.rrr()
    cc2.rrr()
    rr2.rrr()
    fc2.rrr()
    algos.rrr()
    # gui
    guitools.rrr()
    guifront.rrr()
    guiback.rrr()
    # viz
    viz.rrr()
    interact.rrr()
    df2.rrr()
    print('---------------------------')
    print('df2 reset')
    df2.reset()
    print('---------------------------')
    print('[dev] finished dev_reload()')
    print('===========================')
