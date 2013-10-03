from __init__ import *
import os

#db_dir = params.JAG_KELLY
db_dir = params.NAUTS

hs = ld2.HotSpotter()
hs.load_all(db_dir, matcher=False)

vsmany_index = mc2.precompute_index_vsmany(hs)
#
ax2_cx = vsmany_index.ax2_cx
ax2_fx = vsmany_index.ax2_fx
ax2_desc = vsmany_index.ax2_desc
num_agg = len(ax2_cx)
vsmany_flann = vsmany_index.vsmany_flann
k_vsmany = params.__VSMANY_K__+1
checks = params.VSMANY_FLANN_PARAMS['checks']
# Find each databases descriptor K nearest neighbors
def flann_query(desc1):
    # Indexed features
    print('There are %r aggregate features' % num_agg)
    print('Query K=%r nearest neighbors' % k_vsmany+1)
    (qax2_ax, qax2_dists) = vsmany_flann.nn_index(ax2_desc, k_vsmany+1, checks=checks)
    print('...done')
    qax2_cx = ax2_cx[qax2_ax[:, 0:k_vsmany]]
    qax2_fx = ax2_fx[qax2_ax[:, 0:k_vsmany]]
    return (qax2_ax, qax2_dists, qax2_cx, qax2_fx)
(qax2_ax, qax2_dists, qax2_cx, qax2_fx) = flann_query(ax2_desc)


# Each query should find itself as its nearest neighbor
nn_miss = np.where(qax2_ax[:,0] != np.arange(0,len(qax2_ax)))[0]
print('There were at least %r ANN failures.' % len(nn_miss))

# Build a graph of features
import networkx as nx
import matplotlib.pyplot as plt

# Build edges and nodes as the features
nodes = [ax for ax in xrange(num_agg)]
edges = [(ax1, ax2, w) for ax1 in xrange(num_agg) 
         for ax2, w in zip(qax2_ax[ax1], qax2_dists[ax1])]

cx_nodes = np.unique(ax2_cx[nodes])
cx_edges = [(ax2_cx[ax1], ax2_cx[ax2], w) for (ax1, ax2, w) in edges]

print('len(nodes) = %r' % len(nodes))
print('len(edges) = %r' % len(edges))

print('Creating Graph')
MG=nx.MultiDiGraph()
MG.add_weighted_edges_from(cx_edges)
MG.degree(weight='weight')

print('Write to Graphviz format')
nx.write_dot(MG, 'graph.dot')
os.system('dot -Tpng graph.dot -o graph.png')

print('Drawing Graph')
nx.draw(MG)
#dot -Tpng input.dot


def graph_query(hs, qcx):
    desc1 = hs.feats.cx2_desc[qcx]

