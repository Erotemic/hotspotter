#!/usr/env python
from __future__ import division, print_function
import multiprocessing
from hotspotter import encounter
from hscom import helpers
from hsviz import draw_func2 as df2
from hsdev import test_api

USE_TESTCACHE = True


def test_encounter(hs):
    exec(open('hotspotter/encounter.py').read())
    encounter.rrr()
    try:
        if USE_TESTCACHE:
            raise KeyError('use_testcache=False')
        ex2_cxs = helpers.load_testdata('ex2_cxs', uid=hs.get_db_name())
    except KeyError:
        ex2_cxs = encounter.get_chip_encounters(hs)
        helpers.stash_testdata('ex2_cxs', uid=hs.get_db_name())
    cxs = ex2_cxs[-1]
    assert len(cxs) > 1
    qcx2_res = encounter.intra_query_cxs(hs, cxs)
    # Make a graph between the chips
    graph = encounter.make_chip_graph(qcx2_res)
    encounter.viz_chipgraph(hs, graph, fnum=20, with_images=False)
    #encounter.viz_chipgraph(hs, graph, fnum=20, with_images=True)
    df2.update()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    hs = test_api.main()
    test_encounter(hs)
    exec(df2.present())
'''
python _tests/test_encounter.py --dbdir ~/data/work/MISC_Jan12
python _tests/test_encounter.py --dbdir ~/data/work/NAUTS_Dan
'''
