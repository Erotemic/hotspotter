#from os.path import expanduser; f = open(expanduser('~/code/hotspotter/gephi_script.py')); exec(f.read()); f.close()
#

'''
(<type 'org.gephi.filters.FilterControllerImpl'>, 'FilterController')
(<type 'java.lang.Class'>, 'ClockwiseRotate')
(<type 'java.lang.Class'>, 'Contract')
(<type 'java.lang.Class'>, 'CounterClockwiseRotate')
(<type 'java.lang.Class'>, 'Expand')
(<type 'java.lang.Class'>, 'ForceAtlas')
(<type 'java.lang.Class'>, 'ForceAtlas2')
(<type 'java.lang.Class'>, 'FruchtermanReingold')
(<type 'java.lang.Class'>, 'LabelAdjust')
(<type 'java.lang.Class'>, 'Lookup')
(<type 'java.lang.Class'>, 'RandomLayout')
(<type 'java.lang.Class'>, 'YifanHu')
(<type 'java.lang.Class'>, 'YifanHuMultiLevel')
(<type 'java.lang.Class'>, 'YifanHuProportional')
(<type 'function'>, 'addFilter')
(<type 'function'>, 'add_filter')
(<type 'function'>, 'center')
(<type 'function'>, 'exportGraph')
(<type 'function'>, 'getEdgeAttributes')
(<type 'function'>, 'getLayoutBuilders')
(<type 'function'>, 'getNodeAttributes')
(<type 'function'>, 'get_edge_attributes')
(<type 'function'>, 'get_layout_builders')
(<type 'function'>, 'get_node_attributes')
(<type 'function'>, 'importGraph')
(<type 'function'>, 'resetSelection')
(<type 'function'>, 'runLayout')
(<type 'function'>, 'run_layout')
(<type 'function'>, 'selectEdges')
(<type 'function'>, 'selectNodes')
(<type 'function'>, 'selectSubGraph')
(<type 'function'>, 'selectSubgraph')
(<type 'function'>, 'select_edges')
(<type 'function'>, 'select_nodes')
(<type 'function'>, 'select_sub_graph')
(<type 'function'>, 'select_subgraph')
(<type 'function'>, 'setVisible')
(<type 'function'>, 'set_visible')
(<type 'function'>, 'stopLayout')
(<type 'function'>, 'stop_layout')
(<type 'classobj'>, 'Console')
(<type 'javapackage'>, 'java')
(<type 'javapackage'>, 'org')
(<type 'org.gephi.scripting.wrappers.GyGraph'>, 'g')
(<type 'org.gephi.scripting.wrappers.GyGraph'>, 'graph')
(<type 'instance'>, 'console')
(<type 'org.gephi.io.importer.impl.ImportControllerImpl'>, 'ImportController')
(<type 'org.gephi.layout.LayoutControllerImpl'>, 'LayoutController')
(<type 'org.gephi.project.impl.ProjectControllerImpl'>, 'ProjectController')
(<type 'org.gephi.io.exporter.impl.ExportControllerImpl'>, 'ExportController')
(<type 'org.gephi.visualization.VizController'>, 'VizController')
'''
#from os.path import *; execfile(expanduser('~/code/hotspotter/gephi_script.py'))

import java
from os.path import *  # NOQA


def rrr():
    execfile(expanduser('~/code/hotspotter/gephi_script.py'))

if not 'DEVLOCALS' in vars():
    DEVLOCALS = {}
    DEVLOCALS['ISINIT'] = False


def print_available():
    local_items = locals().items()
    item_tuples = []
    for key, val in local_items:
        if not isinstance(val, java.awt.Color):
            item_tuples += [(type(val), key)]
    sorted_items = sorted(item_tuples)
    print('\n'.join(map(str, sorted_items)))

# Useful variables
#ProjectController.newProject()

# pc = Lookup.getDefault().lookup(ProjectController.class)
pc = ProjectController
viz = VizController
lc = LayoutController

proj = pc.getCurrentProject()
wspace = pc.getCurrentWorkspace()
#wspace.init(proj)

#pc.cleanWorkspace(wspace)
cfg = viz.getVizConfig()
vizmodel = viz.getVizModel()
text_model = vizmodel.getTextModel()
graphio = viz.getGraphIO()
modelClassLib = viz.getModelClassLibrary()


def get_gml_fpath():
    #gml_fname = 'HSDB_zebra_with_mothers_cgraph_netx.gml'
    gml_fname = 'GZ_ALL_cgraph_netx.gml'
    gml_dir = expanduser('~/code/hotspotter/graphs/')
    gml_fpath = join(gml_dir, gml_fname)
    return gml_fpath


def ImportCustomGML():
    global DEVLOCALS
    if DEVLOCALS['ISINIT'] is False:
        print('ImportCustomGML() ... opening')
        importGraph(get_gml_fpath())
        DEVLOCALS['ISINIT'] = True
        return True
    else:
        print('ImportCustomGML() ... already open')
        return False
    #gml_file = java.io.File(get_gml_fpath())
    #containter = ImportController.importFile(gml_file)
    #ImportController.process(containter)  # returns void


def centerOnGraph():
    import org.gephi.visualization.VizController as VIZ
    viz = VIZ.getInstance()
    #import org.gephi.project.api.ProjectController as pc
    viz.getGraphIO().centerOnGraph()
    viz.getEngine().getScheduler().requireUpdateVisible()


def SetCustomVizParams():
    print('SetCustomVizParams()')
    # No Hulls
    viz.vizModel.setShowHulls(False)
    # Smaller edges
    vizmodel.setEdgeScale(.2)
    # Dark background
    vizmodel.setBackgroundColor(black)
    # View the name of the animal nodes
    node_attrs = getNodeAttributes()
    if 'name' in node_attrs:
        name_attr = node_attrs['name']
        name_col = name_attr.underlyingAttributeColumn
        viz.textManager.model.nodeTextColumns[0] = name_col
    # Show animal names
    text_model.setShowNodeLabels(True)
    # Smaller font
    try:
        smallfont = java.awt.Font('Mono Dyslexic', 0, 8)
        text_model.setNodeFont(smallfont)
    except Exception:
        print(ex)
        smallfont = java.awt.Font('Arial', 0, 8)
        text_model.setNodeFont(smallfont)
    #text_model.setNodeColor(orange)


def CustomForceParamsBAD(layout):
    layout.maxDisplacement = 1e2
    layout.repulsionStrength = 1e3
    layout.attractionStrength = 1e-3
    layout.gravity = 1e3
    layout.freezeStrength = 1e2  # autostab strength
    layout.freezeInertia = 1e2  # Autostab sensitivity
    layout.cooling = 1e0
    layout.inertia = 1e0
    layout.speed = 1e2

def CustomForceParams(layout):
    layout.maxDisplacement = 1e1
    layout.repulsionStrength = 1e3
    layout.attractionStrength = 1e-1
    layout.gravity = -1e3
    layout.freezeStrength = 1e2  # autostab strength
    layout.freezeInertia = 1e0  # Autostab sensitivity
    layout.cooling = 1e0
    layout.inertia = 1e0
    layout.speed = 1e0

if ImportCustomGML():
    SetCustomVizParams()
else:
    print('stopping layout')
    stopLayout()

#stopLayout()
forces_ = ForceAtlas()
layout  = forces_.buildLayout()

selected_layout = lc.model.getSelectedLayout()
print('selected_layout: %r' % selected_layout)
print('  custom_layout: %r' % layout)
CustomForceParams(layout)
lc.setLayout(layout)
#lc.model.loadProperties(layout)
print('Running Layout')
#run_layout(layout.getBuilder, iters=100)
print('Done')
#centerOnGraph()
#stopLayout()

#text_model.setTextColumns()
#ImportController.getFileImporter(gml_fpath)
# Viz Config


def interpolateColor(alpha, color1=blue, color2=orange):
    assert alpha >= 0 and alpha <= 1
    red   = int(alpha * color1.red   + (1.0 - alpha) * color2.red)
    green = int(alpha * color1.green + (1.0 - alpha) * color2.green)
    blue  = int(alpha * color1.blue  + (1.0 - alpha) * color2.blue)
    color3 = java.awt.Color(red, green, blue)
    return color3


def CustomEdgeColors():
    print('Getting min max weight')
    minWeight = 1e29
    maxWeight = -1e29

    for edge in graph.edges:
        if edge.weight < minWeight:
            minWeight = edge.weight
        if edge.weight > maxWeight:
            maxWeight = edge.weight
    print('minWeight, maxWeight = %r, %r' % (minWeight, maxWeight))
    scale = 'log10'
    if scale == 'log10':
        from math import log10
        scalefn = lambda num: log10(num)
    if scale == 'log':
        from math import log
        scalefn = lambda num: log(num)
    elif scale == 'linear':
        scalefn = lambda num: num
    shift2 = .3
    min_ = scalefn(minWeight)
    max_ = scalefn(maxWeight)
    max_ = (shift2 * (max_ - min_)) + min_
    range_ = max_ - min_
    def interpolateEdge(edge):
        alpha = (scalefn(edge.weight) - min_) / range_
        alpha = min(1, max(0, alpha))
        color = interpolateColor(alpha, blue, orange)
        edge.color = color
    print('Setting Edge Colors')
    map(interpolateEdge, graph.edges)
    print('Done')


CustomEdgeColors()


def doForce():
    #org.gephi.layout.plugin.forceAtlas.
    import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2Builder
    runLayout(org.gephi.layout.plugin.forceAtlas2.ForceAtlas2Builder)
    stopLayout()
