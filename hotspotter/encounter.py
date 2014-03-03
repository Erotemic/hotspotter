from __future__ import print_function, division
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile,
 printDBG) = __common__.init(__name__, '[encounter]', DEBUG=False)
# Python
from itertools import izip
# Science
import networkx as netx
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
# HotSpotter
from hotspotter import match_chips3 as mc3
from hscom import fileio as io
from hscom import helpers as util


def compute_encounters(hs, seconds_thresh=15):
    '''
    clusters encounters togethers (by time, not space)
    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.  Animals are identified within each encounter.
    '''
    if not 'seconds_thresh' in vars():
        seconds_thresh = 3

    # For each image
    gx_list = hs.get_valid_gxs()

    # TODO: Get image GPS location
    #gps_info_list = hs.gx2_exif(gx_list, tag='GPSInfo')
    #gps_lat_list = hs.gx2_exif(gx_list, tag='GPSLatitude')
    #gps_lon_list = hs.gx2_exif(gx_list, tag='GPSLongitude')
    #gps_latref_list = hs.gx2_exif(gx_list, tag='GPSLatitudeRef')
    #gps_lonref_list = hs.gx2_exif(gx_list, tag='GPSLongitudeRef')

    # Get image timestamps
    datetime_list = hs.gx2_exif(gx_list, tag='DateTime')

    nImgs = len(datetime_list)
    valid_listx = [ix for ix, dt in enumerate(datetime_list) if dt is not None]
    nWithExif = len(valid_listx)
    nWithoutExif = nImgs - nWithExif
    print('[encounter] %d / %d images with exif data' % (nWithExif, nImgs))
    print('[encounter] %d / %d images without exif data' % (nWithoutExif, nImgs))

    # Convert datetime objects to unixtime scalars
    unixtime_list = [io.exiftime_to_unixtime(datetime_str) for datetime_str in datetime_list]
    unixtime_list = np.array(unixtime_list)

    # Agglomerative clustering of unixtimes
    print('[encounter] clustering')
    X_data = np.vstack([unixtime_list, np.zeros(len(unixtime_list))]).T
    gx2_clusterid = fclusterdata(X_data, seconds_thresh, criterion='distance')

    # Reverse the image to cluster index mapping
    clusterx2_gxs = [[] for _ in xrange(gx2_clusterid.max())]
    for gx, clusterx in enumerate(gx2_clusterid):
        clusterx2_gxs[clusterx - 1].append(gx)  # IDS are 1 based

    # Print images per encouter statistics
    clusterx2_nGxs = np.array(map(len, clusterx2_gxs))
    print('[encounter] image per encounter stats:\n %s'
          % util.pstats(clusterx2_nGxs, True))

    # Sort encounters by images per encounter
    ex2_clusterx = clusterx2_nGxs.argsort()
    gx2_ex  = [None] * len(gx2_clusterid)
    ex2_gxs = [None] * len(ex2_clusterx)
    for ex, clusterx in enumerate(ex2_clusterx):
        gxs = clusterx2_gxs[clusterx]
        ex2_gxs[ex] = gxs
        for gx in gxs:
            gx2_ex[gx] = ex
    return gx2_ex, ex2_gxs


def build_encounter_ids(ex2_gxs, gx2_clusterid):
    USE_STRING_ID = True
    gx2_eid = [None] * len(gx2_clusterid)
    for ex, gxs in enumerate(ex2_gxs):
        for gx in gxs:
            nGx = len(gxs)
            gx2_eid[gx] = ('ex=%r_nGxs=%d' % (ex, nGx)
                           if USE_STRING_ID else
                           ex + (nGx / 10 ** np.ceil(np.log(nGx) / np.log(10))))


def get_chip_encounters(hs):
    gx2_ex, ex2_gxs = compute_encounters(hs)
    # Build encounter to chips from encounter to images
    ex2_cxs = [None for _ in xrange(len(ex2_gxs))]
    for ex, gxs in enumerate(ex2_gxs):
        ex2_cxs[ex] = util.flatten(hs.gx2_cxs(gxs))
    # optional
    # resort encounters by number of chips
    ex2_nCxs = map(len, ex2_cxs)
    ex2_cxs = [y for (x, y) in sorted(zip(ex2_nCxs, ex2_cxs))]
    return ex2_cxs


def get_fmatch_iter(res):
    # USE res.get_fmatch_iter()
    fmfsfk_enum = enumerate(izip(res.cx2_fm, res.cx2_fs, res.cx2_fk))
    fmatch_iter = ((cx, fx_tup, score, rank)
                   for cx, (fm, fs, fk) in fmfsfk_enum
                   for (fx_tup, score, rank) in izip(fm, fs, fk))
    return fmatch_iter


def get_cxfx_enum(qreq):
    ax2_cxs = qreq._data_index.ax2_cx
    ax2_fxs = qreq._data_index.ax2_fx
    cxfx_enum = enumerate(izip(ax2_cxs, ax2_fxs))
    return cxfx_enum


def intra_query_cxs(hs, cxs):
    dcxs = qcxs = cxs
    qreq = mc3.prep_query_request(qreq=hs.qreq, qcxs=qcxs, dcxs=dcxs,
                                  query_cfg=hs.prefs.query_cfg)
    qcx2_res = mc3.process_query_request(hs, qreq)
    return qcx2_res


#def intra_encounter_match(hs, cxs, **kwargs):
    # Make a graph between the chips
    #qcx2_res = intra_query_cxs(cxs)
    #graph = make_chip_graph(qcx2_res)
    # TODO: Make a super cool algorithm which does this correctly
    #graph.cutEdges(**kwargs)
    # Get a temporary name id
    # TODO: ensure these name indexes do not conflict with other encounters
    #cx2_nx, nx2_cxs = graph.getConnectedComponents()
    #return graph


def execute_all_intra_encounter_match(hs, **kwargs):
    # Group images / chips into encounters
    ex2_cxs = get_chip_encounters(hs)
    # For each encounter
    ex2_names = {}
    for ex, cxs in enumerate(ex2_cxs):
        pass
        # Perform Intra-Encounter Matching
        #nx2_cxs = intra_encounter_match(hs, cxs)
        #ex2_names[ex] = nx2_cxs
    return ex2_names


def inter_encounter_match(hs, eid2_names=None, **kwargs):
    # Perform Inter-Encounter Matching
    #if eid2_names is None:
        #eid2_names = intra_encounter_match(hs, **kwargs)
    all_nxs = util.flatten(eid2_names.values())
    for nx2_cxs in eid2_names:
        qnxs = nx2_cxs
        dnxs = all_nxs
        name_result = hs.query(qnxs=qnxs, dnxs=dnxs)
    qcx2_res = name_result.chip_results()
    graph = netx.Graph()
    graph.add_nodes_from(range(len(qcx2_res)))
    graph.add_edges_from([res.cx2_fm for res in qcx2_res.itervalues()])
    graph.setWeights([(res.cx2_fs, res.cx2_fk) for res in qcx2_res.itervalues()])
    graph.cutEdges(**kwargs)
    cx2_nx, nx2_cxs = graph.getConnectedComponents()
    return cx2_nx


def print_encounter_stats(ex2_cxs):
    ex2_nCxs = map(len, ex2_cxs)
    ex_statstr = util.printable_mystats(ex2_nCxs)
    print('num_encounters = %r' % len(ex2_nCxs))
    print('encounter_stats = %r' % (ex_statstr,))
