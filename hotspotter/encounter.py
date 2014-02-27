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
from hsviz import draw_func2 as df2


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


def make_feature_graph(qreq, qcx2_res, use_networkx=True):
    # Make a graph between the chips
    cxfx2_ax = {(cx, fx): ax for ax, (cx, fx) in get_cxfx_enum(qreq)}
    def w_edge(cx1, cx2, fx1, fx2, score, rank):
        ax1 = cxfx2_ax[(cx1, fx1)]
        ax2 = cxfx2_ax[(cx2, fx2)]
        attr_dict =  {'score': score, 'rank': rank}
        return (ax1, ax2, attr_dict)
    nodes = [(ax, {'fx': fx, 'cx': cx}) for ax, (cx, fx) in get_cxfx_enum(qreq)]
    weighted_edges = [w_edge(cx1, cx2, fx1, fx2, score, rank)
                      for (cx1, res) in qcx2_res.iteritems()
                      for (cx2, (fx1, fx2), score, rank) in get_fmatch_iter(res)
                      if score > 0]
    if use_networkx:
        graph = netx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(weighted_edges)
    else:
        vx2_ax = cxfx2_ax.values()
        import graph_tool
        graph = graph_tool.Graph(g=None, directed=True, prune=False, vorder=None)
        vertex_list = graph.add_vertex(n=len(nodes))

        v_fx = graph.new_vertex_property("int")
        v_cx = graph.new_vertex_property("int")

        e_score = graph.new_edge_property("float")
        e_rank = graph.new_edge_property("int")

        for v, (ax, vprops) in zip(vertex_list, nodes):
            v_cx[v] = int(vprops['cx'])
            v_fx[v] = int(vprops['fx'])

        mark_prog, end_prog = util.progress_func(len(weighted_edges))
        count = 0
        for ax1, ax2, prop_dict in weighted_edges:
            mark_prog(count)
            count += 1
            vx1 = vx2_ax.index(ax1)
            vx2 = vx2_ax.index(ax2)
            v1 = graph.vertex(vx1)
            v2 = graph.vertex(vx2)
            e = graph.add_edge(v1, v2)
            e_score[e] = float(prop_dict['score'])
            e_rank[e] = int(prop_dict['rank'])
        mark_prog(count)
        end_prog()
        #import graph_tool.draw

        graph.save('test_graph.dot')
    return graph


def make_chip_graph(qcx2_res):
    # Make a graph between the chips
    nodes = qcx2_res.keys()
    #attr_edges = [(res.qcx, cx, {'score': score})
                    #for res in qcx2_res.itervalues()
                    #for cx, score in enumerate(res.cx2_score) if score > 0]
    weighted_edges = [(res.qcx, cx, score)
                      for res in qcx2_res.itervalues()
                      for cx, score in enumerate(res.cx2_score) if score > 0]
    graph = netx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(weighted_edges)
    return graph


def viz_graph(graph):
    netx.draw(graph)


def viz_chipgraph(hs, graph, fnum=1, with_images=False):
    # Adapated from
    # https://gist.github.com/shobhit/3236373
    print('[encounter] drawing chip graph')
    df2.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = df2.gca()
    #pos = netx.spring_layout(graph)
    pos = netx.graphviz_layout(graph)
    netx.draw(graph, pos=pos, ax=ax)
    if with_images:
        cx_list = graph.nodes()
        pos_list = [pos[cx] for cx in cx_list]
        thumb_list = hs.get_thumb(cx_list, 16, 16)
        draw_images_at_positions(thumb_list, pos_list)


def draw_images_at_positions(img_list, pos_list):
    print('[encounter] drawing %d images' % len(img_list))
    # Thumb stack
    ax  = df2.gca()
    fig = df2.gcf()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    mark_progress, end_progress = util.progress_func(len(pos_list), lbl='drawing img')
    for ix, ((x, y), img) in enumerate(izip(pos_list, img_list)):
        mark_progress(ix)
        xx, yy = trans((x, y))  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        #
        width, height = img.shape[0:2]
        tlx = xa - (width / 2.0)
        tly = ya - (height / 2.0)
        img_bbox = [tlx, tly, width, height]
        # Make new axis for the image
        img_ax = df2.plt.axes(img_bbox)
        img_ax.imshow(img)
        img_ax.set_aspect('equal')
        img_ax.axis('off')
    end_progress()


def intra_query_cxs(hs, cxs):
    dcxs = qcxs = cxs
    qreq = mc3.prep_query_request(qreq=hs.qreq,
                                  qcxs=qcxs,
                                  dcxs=dcxs,
                                  query_cfg=hs.prefs.query_cfg)
    qcx2_res = mc3.process_query_request(hs, qreq)
    return qcx2_res


def intra_encounter_match(hs, cxs, **kwargs):
    # Make a graph between the chips
    qcx2_res = intra_query_cxs(cxs)
    graph = make_chip_graph(qcx2_res)
    # TODO: Make a super cool algorithm which does this correctly
    #graph.cutEdges(**kwargs)
    # Get a temporary name id
    # TODO: ensure these name indexes do not conflict with other encounters
    #cx2_nx, nx2_cxs = graph.getConnectedComponents()
    return graph


def execute_all_intra_encounter_match(hs, **kwargs):
    # Group images / chips into encounters
    ex2_cxs = get_chip_encounters(hs)
    # For each encounter
    ex2_names = {}
    for ex, cxs in enumerate(ex2_cxs):
        pass
        # Perform Intra-Encounter Matching
        nx2_cxs = intra_encounter_match(hs, cxs)
        ex2_names[ex] = nx2_cxs
    return ex2_names


def inter_encounter_match(hs, eid2_names=None, **kwargs):
    # Perform Inter-Encounter Matching
    if eid2_names is None:
        eid2_names = intra_encounter_match(hs, **kwargs)
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
