#!/usr/bin/env python
from __future__ import division, print_function
import __builtin__
import multiprocessing
from hotspotter import encounter
from hscom import util
from hsviz import draw_func2 as df2
from hsdev import test_api
from hsdev import dev_augmenter
from hscom import hsgraph
import networkx as netx  # NOQA
import sys
from os.path import join

USE_TESTCACHE = True

INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    __builtin__.print('\n=============================')
    __builtin__.print(msg)
    if INTERACTIVE:
        raw_input('press enter to continue')

if __name__ == '__main__':
    multiprocessing.freeze_support()

    print('[TEST] END TEST')
    hs, back, app, is_root = test_api.main_init(preload=True)

    encounter.rrr()
    dev_augmenter.dev_reload.reload_all_modules()
    try:
        if USE_TESTCACHE:
            raise KeyError('use_testcache=False')
        ex2_cxs = util.load_testdata('ex2_cxs', uid=hs.get_db_name())
    except KeyError:
        # Build encounter clusters
        ex2_cxs = encounter.get_chip_encounters(hs)
        util.save_testdata('ex2_cxs', uid=hs.get_db_name())
    encounter.print_encounter_stats(ex2_cxs)
    cxs = ex2_cxs[-1]
    assert len(cxs) > 1

    # Build result list
    from hotspotter import match_chips3 as mc3
    mc3.rrr()
    qreq = mc3.quickly_ensure_qreq(hs, qcxs=cxs, dcxs=cxs)
    # Query within an encounter
    qcx2_res = mc3.bigcache_query(hs, qreq, batch_size=64)
    #encounter.intra_query_cxs(hs, cxs)

    # Use result list to build matching graph
    hsgraph.rrr()
    util.rrr()
    graphs_dir = 'graphs'
    util.ensuredir(graphs_dir)

    #cgraph_gtool = hsgraph.make_chip_graph(hs, qcx2_res, 'graph-tool')
    #fgraph_gtool = hsgraph.make_feature_graph(hs, qcx2_res, 'graph-tool')

    cgraph_netx  = hsgraph.make_chip_graph(hs, qcx2_res, 'netx')
    cgraph_fpath = join(graphs_dir, hs.get_db_name() + '_cgraph')
    hsgraph.export(cgraph_netx,  cgraph_fpath + '_netx', 'gml')

    #fgraph_fpath = join(graphs_dir, hs.get_db_name() + '_fgraph')
    #fgraph_netx  = hsgraph.make_feature_graph(hs, qcx2_res, 'netx')
    #hsgraph.export(fgraph_netx,  fgraph_fpath + '_netx', 'gml')

    #hsgraph.export(cgraph_gtool, cgraph_fpath + '_gtool', 'gml')
    #dot_fpath = hsgraph.export_dotfile(fgraph_netx,  'graphs/fgraph_netx')
    #hsgraph.render_graph(cgraph_gtool, 'graphs/cgraph_gtool')
    #hsgraph.render_graph(fgraph_gtool, 'graphs/fgraph_gtool')
    #hsgraph.show_graph(graph_netx)

    try:
        pass
        #import graph_tool
        #graph = graph_tool.Graph()
    except ImportError as ex:
        print(ex)
    #encounter.viz_chipgraph(hs, graph, fnum=20, with_images=False)
    #encounter.viz_chipgraph(hs, graph, fnum=20, with_images=True)

    df2.update()
    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
