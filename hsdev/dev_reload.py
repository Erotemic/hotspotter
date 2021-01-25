
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[dev_reload]')


def reload_all_modules():
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
    from hscom import Parallelize as parallel
    #from hscom import Preferences as prefs
    #from hscom import Printable
    #from hscom import argparse2
    from hscom import cross_platform as cplat
    from hscom import fileio as io
    from hscom import helpers as util
    from hscom import latex_formater
    from hscom import params
    from hscom import tools
    # Viz
    from hsviz import draw_func2 as df2
    from hsviz import extract_patch
    from hsviz import viz
    from hsviz import interact
    from hsviz import allres_viz
    # GUI
    from hsgui import guitools
    from hsgui import guifront
    from hsgui import guiback
    # DEV
    from . import dev_stats
    from . import dev_consistency
    from . import dev_api
    from . import dev_reload
    # Self
    rrr()
    # com
    util.rrr()
    io.rrr()
    cplat.rrr()
    parallel.rrr()
    #prefs.rrr()
    #Printable.rrr()
    #argparse2.rrr()
    latex_formater.rrr()
    params.rrr()
    tools.rrr()
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
    extract_patch.rrr()
    viz.rrr()
    interact.rrr()
    df2.rrr()
    allres_viz.rrr()
    # dev
    dev_stats.rrr()
    dev_consistency.rrr()
    dev_api.rrr()
    dev_reload.rrr()

    print('---------------------------')
    print('df2 reset')
    df2.reset()
    print('---------------------------')
    print('[dev] finished dev_reload()')
    print('===========================')
