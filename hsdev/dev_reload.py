from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[dev_reload]', DEBUG=False)
import imp


def reload_all_modules():
    print('===========================')
    print('[dev] performing dev_reload')
    print('---------------------------')
    from hsapi import DataStructures as ds
    from hsapi import algos
    from hsapi import load_data2 as ld2
    from hsapi import chip_compute2 as cc2
    from hsapi import feature_compute2 as fc2
    from hsapi import match_chips3 as mc3
    from hsapi import matching_functions as mf
    from hsapi import nn_filters
    from hsapi import report_results2 as rr2
    from hsapi import voting_rules2 as vr2
    # Common
    from hscom import Parallelize as parallel
    #from hscom import Preferences as prefs
    #from hscom import Printable
    #from hsdev import argparse2
    from hscom import cross_platform as cplat
    from hscom import fileio as io
    from hscom import util
    from hscom import latex_formater
    from hsdev import params
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
    from hsdev import dev_stats
    from hsdev import dev_consistency
    from hsdev import dev_augmenter
    from hsdev import dev_reload
    # Vtool
    import vtool
    import vtool.patch as ptool
    import vtool.linalg as ltool
    import vtool.keypoint as ktool
    import vtool.drawtool as dtool
    import vtool.histogram as htool

    imp.reload(vtool)
    imp.reload(ptool)
    imp.reload(ktool)
    imp.reload(dtool)
    imp.reload(htool)
    imp.reload(ltool)

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
    dev_augmenter.rrr()
    dev_reload.rrr()

    print('---------------------------')
    print('df2 reset')
    df2.reset()
    print('---------------------------')
    print('[dev] finished dev_reload()')
    print('===========================')
