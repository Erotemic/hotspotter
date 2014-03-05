from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[graph]', DEBUG=False)
from hscom import helpers as util
from os.path import splitext, exists
import cross_platform as cplat
import networkx as netx
from itertools import izip
from hsviz import draw_func2 as df2
try:
    import graph_tool as gtool
except ImportError as ex:
    gtool = None
    print(ex)


def show_graph(graph, **kwargs):
    img_fpath = render_graph(graph, **kwargs)
    print('[graph] show_graph()')
    util.show_img_fpath(img_fpath)
    print('[graph] finished show')


def render_graph(graph, fpath='graph'):
    print('[graph] render_graph()')
    dot_fpath = export_dot(graph, fpath)
    img_fpath = export_dot(dot_fpath)
    return img_fpath


def export_dot(graph, fpath='graph'):
    print('[graph] export_dotfile()')
    dot_fpath = util.ensure_ext(fpath, '.dot')
    if isinstance(graph, netx.Graph):
        netx.write_dot(graph, dot_fpath)
    elif isinstance(graph, gtool.Graph):
        graph.save(dot_fpath, fmt='dot')
    return dot_fpath


def export_gephi(graph, fpath):
    gexf_fpath = util.ensure_ext(fpath, '.gexf')
    if isinstance(graph, netx.Graph):
        netx.write_gexf(graph, gexf_fpath)
    elif isinstance(graph, gtool.Graph):
        raise NotImplementedError()
    return gexf_fpath


def export_gml(graph, fpath):
    fpath = util.ensure_ext(fpath, '.gml')
    if isinstance(graph, netx.Graph):
        netx.write_gml(graph, fpath)
    elif isinstance(graph, gtool.Graph):
        graph.save(fpath, fmt='gml')
    else:
        raise NotImplementedError()
    return fpath


def export_xml(graph, fpath):
    fpath = util.ensure_ext(fpath, '.xml')
    graph.save(fpath, fmt='xml')


def export(graph, fpath, fmt):
    print('[graph] exporting graph: %r' % fpath)
    if fmt == 'gml':
        export_gml(graph, fpath)
    if fmt == 'dot':
        export_dot(graph, fpath)


#def export_json(graph, fpath):
    #json_fpath = util.ensure_ext(fpath, '.json')
    #if isinstance(graph, netx.Graph):
        #from networkx.readwrite import json_graph
        #json_graph.dumps(graph)


def render_dotfile(dot_fpath, img_fpath=None, img_ext='.jpg'):
    print('[graph] render_dotfile()')
    if img_fpath is None:
        img_fpath = splitext(dot_fpath)[0] + img_ext
    else:
        img_ext = splitext(img_fpath)[1]
    dot_fmt = '-T' + img_ext[1:]
    cplat._cmd('dot', dot_fmt, dot_fpath, '-o', img_fpath)
    print('[graph] finished rendering dotfile')
    if not exists(img_fpath):
        raise OSError('cannot render dotfile')
    return img_fpath


def viz_netx_chipgraph(hs, graph, fnum=1, with_images=False):
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
        netx_draw_images_at_positions(thumb_list, pos_list)


def netx_draw_images_at_positions(img_list, pos_list):
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


@util.indent_decor('[feat_graph]')
def make_feature_graph(hs, qcx2_res, *args, **kwargs):
    qreq = hs.qreq
    # Make a graph between the chips
    cxfx2_ax = {(cx, fx): ax for ax, (cx, fx) in qreq.get_cxfx_enum()}
    # Build feature nodes
    nodes = [(ax, fx, cx)
             for ax, (cx, fx) in qreq.get_cxfx_enum()]
    # Build feature edges
    edges = [(cxfx2_ax[(cx1, fx1)], cxfx2_ax[(cx2, fx2)], score, rank)
             for (cx1, res) in qcx2_res.iteritems()
             for (cx2, (fx1, fx2), score, rank) in res.get_fmatch_iter()
             if score > 0]
    node_lbls = [('fx', 'int'), ('cx', 'int')]
    edge_lbls = [('weight', 'float'), ('rank', 'int')]
    return make_graph(nodes, edges, node_lbls, edge_lbls, *args, **kwargs)


@util.indent_decor('[chip_graph]')
def make_chip_graph(hs, qcx2_res, *args, **kwargs):
    # Make a graph between the chips

    nodes = [(cx, hs.tables.cx2_cid[cx], hs.tables.cx2_nx[cx], hs.cx2_name(cx)) for cx in qcx2_res.iterkeys()]
    edges = [(res.qcx, cx, score)
             for res in qcx2_res.itervalues()
             for cx, score in enumerate(res.cx2_score) if score > 0]
    node_lbls = [('cid', 'int'), ('nx', 'int'), ('name', 'int')]
    edge_lbls = [('weight', 'float')]
    return make_graph(nodes, edges, node_lbls, edge_lbls, *args, **kwargs)


def make_graph(nodes, edges, node_lbls, edge_lbls, graphlib='networkx'):
    print('[graph] --- make %s graph ---' % graphlib)
    with util.Timer('timing make %s graph' % graphlib):
        nNodes = len(nodes)
        nEdges = len(edges)
        #nElements = nNodes + nEdges
        print('[graph] nNodes, nEdges = (%r, %r)' % (nNodes, nEdges,))
        # Attempts to generically make graph
        if graphlib in ['networkx', 'netx']:
            netx_graph = make_netx_graph(nodes, edges, node_lbls, edge_lbls)
            return netx_graph
        elif graphlib in ['graph-tool', 'gtool']:
            gtool_graph = make_gtool_graph(nodes, edges, node_lbls, edge_lbls)
            return gtool_graph
        else:
            raise NotImplementedError(graphlib)


def make_netx_graph(nodes, edges, node_lbls=[], edge_lbls=[]):
    # Make a graph between the chips
    netx_nodes = [(ntup[0], {key[0]: val for (key, val) in izip(node_lbls, ntup[1:])})
                  for ntup in iter(nodes)]
    netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in izip(edge_lbls, etup[2:])})
                  for etup in iter(edges)]
    netx_graph = netx.DiGraph()
    netx_graph.add_nodes_from(netx_nodes)
    netx_graph.add_edges_from(netx_edges)
    return netx_graph


def make_gtool_graph(nodes, edges, node_lbls=[], edge_lbls=[]):
    #print('[graph] new graph')
    gtool_graph = gtool.Graph(g=None, directed=True, prune=False, vorder=None)

    # Graph gtool property structures
    #print('[graph] creating attributes')
    node_props, edge_props = {}, {}
    for lbl, type_ in node_lbls:
        node_props[lbl] = gtool_graph.new_vertex_property(type_)
    for lbl, type_ in edge_lbls:
        edge_props[lbl] = gtool_graph.new_edge_property(type_)

    # Create Vertex List
    print('[graph] adding vertexes')
    vertex_list = gtool_graph.add_vertex(n=len(nodes))

    # Add Vertex Properties
    #print('[graph] adding vertex properties')
    for (vert, ntup) in izip(vertex_list, nodes):
        for lblx, (lbl, type_) in enumerate(node_lbls):
            node_props[lbl][vert] = ntup[lblx + 1]

    # Create Edge List
    print('[graph] adding edges')
    # make lookup structures
    nx2_vx = {ntup[0]: vx for vx, ntup in enumerate(nodes)}
    vx2_vert = gtool_graph.vertex    # alias
    add_edge = gtool_graph.add_edge  # alias
    # map node indexes into vertex indexes
    edge_vxs   = [(nx2_vx[etup[0]], nx2_vx[etup[1]])
                  for etup in iter(edges)]           # nx -> vx
    edge_verts = [(vx2_vert(vx1), vx2_vert(vx2))
                  for (vx1, vx2) in iter(edge_vxs)]  # vx -> vert
    edge_list  = [add_edge(v1, v2)
                  for (v1, v2) in iter(edge_verts)]  # vert -> edges
    # Add Edge Properties
    for (edge, etup) in izip(edge_list, edges):
        for lblx, (lbl, type_) in enumerate(edge_lbls):
            eprop = edge_props[lbl]
            eprop[edge] = etup[lblx + 2]
    return gtool_graph
