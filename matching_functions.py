import itertools
import sys
import os
import warnings
import textwrap
# Hotspotter Frontend Imports
import draw_func2 as df2
# Hotspotter Imports
import fileio as io
import helpers
from helpers import Timer, tic, toc, printWARN
from Printable import DynStruct
import algos
import helpers
import spatial_verification2 as sv2
import load_data2
import params
# Math and Science Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflann
import scipy as sp
import scipy.sparse as spsparse
import sklearn.preprocessing 
from itertools import izip, chain
import investigate_chip as invest

def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()


    
#============================
# Nearest Neighbors
#============================
'''
Filter Example Data:
burst_thresh = 2
qfx2_fx   = np.array([(1, 3, 4), (8, 2, 3), (3, 6, 2), (3, 7, 1), (3, 2, 4), (1, 2, 2)])
qfx2_dist = np.array([(1, 2, 3), (4, 5, 6), (1, 6, 9), (0, 2, 4), (4, 3, 7), (2, 3, 4)])
qfx2_valid = np.ones(qfx2_fx.shape, dtype=np.bool)
'''
# All neighbors are valid
#qfx2_valid = np.ones(qfx2_dx.shape, dtype=np.bool)

def nearest_neighbors(hs, qcxs, data_index, nn_params):
    'Plain Nearest Neighbors'
    K = nn_params.K
    Knorm = nn_params.Knorm
    checks = nn_params.checks
    cx2_desc = hs.feats.cx2_desc
    nn_index = data_index.flann.nn_index
    func = lambda qcx: nn_index(cx2_desc[qcx], K+Knorm, checks=checks)
    qcx2_neighbors = {qcx:func(qcx) for qcx in qcxs}
    return qcx2_neighbors

def _nn_normalized_weight(hs, qcx2_neighbors, data_index, normweight_fn):
    # Only valid for vsone
    K = nn_params.K
    Knorm = nn_params.Knorm
    qcx2_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (_, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nndist = qfx2_dist[:, K:]
        qfx2_normdist = qfx2_dist[:, 0:(K+Knorm)]
        qfx2_ratio = normweight_fn(qfx2_nndist, qfx2_normdist)
        qcx2_weight[qcx] = qfx2_ratio
    return qcx2_weight

eps = 1E-8
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+eps)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+eps)
def LNBNN_fn(vdist, ndist): return (ndist - vdist) / 1000.0

def nn_ratio_weight(hs, qcx2_neighbors, data_index):
    return _nn_normalized_weight(hs, qcx2_neighbors, data_index, RATIO_fn)
def nn_lnbnn_weight(hs, qcx2_neighbors, data_index):
    return _nn_normalized_weight(hs, qcx2_neighbors, data_index, LNBNN_fn)
def nn_lnrat_weight(hs, qcx2_neighbors, data_index):
    return _nn_normalized_weight(hs, qcx2_neighbors, data_index, LNRAT_fn)

def nn_bursty_weight(hs, qcx2_neighbors, data_index, nn_params):
    'Filters matches to a feature which is matched > burst_thresh #times'
    # Half-generalized to vsmany
    # Assume the first nRows-1 rows are the matches (last row is normalizer)
    K = nn_params.K
    Knorm = nn_params.Knorm
    qcx2_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        dx2_frequency  = np.bincount(qfx2_nn.flatten())
        qfx2_bursty = dx2_frequency[qfx2_nn]
        qcx2_weight[qcx] = qfx2_bursty
    return qcx2_weight

def nn_reciprocal_weight(hs, qcx2_neighbors, data_index, nn_params):
    'Filters a nearest neighbor to only reciprocals'
    K = nn_params.K

    Krecip = nn_params.K_reciprocal
    checks = nn_params.checks
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dx2_data = data_index.ax2_data
    data_flann = data_index.flann
    qcx2_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        nQuery = len(qfx2_dx)
        dim = dx2_data.shape[1]
        # Get the original K nearest features
        qx2_nndx = dx2_data[qfx2_dx[:, 0:K]]
        qx2_nndist = qfx2_dist[:, 0:K]
        qx2_nndx.shape = (nQuery*K, dim)
        # TODO: Have the option for this to be both indexes.
        (_nn2_dx, _nn2_dists) = data_flann.nn_index(qx2_nndx, Krecip, checks=checks)
        # Get the maximum distance of the Krecip reciprocal neighbors
        _nn2_dists.shape = (nQuery, K, Krecip)
        qfx2_recipmaxdist = _nn2_dists.max(2)
        # Test if nearest neighbor distance is less than reciprocal distance
        qfx2_reciprocalness = qfx2_recipmaxdist - qx2_nndist
        qcx2_weight[qcx] = qfx2_reciprocalness
    return qcx2_weight

def nn_roidist_weight(hs, qcx2_neighbors, data_index, nn_params):
    'Filters a matches to those within roughly the same spatial arangement'
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts = hs.feats.cx2_kpts
    K = nn_params.K
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    cx2_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:,0:K]
        # Get matched chip sizes #.0300s
        qfx2_kpts = cx2_kpts[qcx]
        nQuery = len(qfx2_dx)
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_chipsize2 = np.array([cx2_rchip_size[cx] for cx in qfx2_cx.flat])
        qfx2_chipsize2.shape = (nQuery, K, 2)
        qfx2_chipdiag2 = np.sqrt((qfx2_chipsize2**2).sum(2))
        # Get query relative xy keypoints #.0160s / #.0180s (+cast)
        qdiag = np.sqrt((np.array(cx2_rchip_size[qcx])**2).sum())
        qfx2_xy1 = np.array(qfx2_kpts[:, 0:2], np.float)
        qfx2_xy1[:,0] /= qdiag
        qfx2_xy1[:,1] /= qdiag
        # Get database relative xy keypoints
        qfx2_xy2 = np.array([cx2_kpts[cx][fx, 0:2] for (cx, fx) in
                            izip(qfx2_cx.flat, qfx2_fx.flat)], np.float)
        qfx2_xy2.shape = (nQuery, K, 2)
        qfx2_xy2[:,:,0] /= qfx2_chipdiag2
        qfx2_xy2[:,:,1] /= qfx2_chipdiag2
        # Get the relative distance # .0010s
        qfx2_K_xy1 = np.rollaxis(np.tile(qfx2_xy1, (K, 1, 1)), 1)
        qfx2_xydist = ((qfx2_K_xy1 - qfx2_xy2)**2).sum(2)
        cx2_weight[qcx] = qfx2_xydist
    return cx2_weight

def nn_scale_weight(hs, qcx2_neighbors, data_index, nn_params):
    # Filter by scale for funzies
    K = nn_params.K
    cx2_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:,0:K]
        nQuery = len(qfx2_dx)
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_det1 = np.array(qfx2_kpts[:, [2,4]], np.float).prod(1)
        qfx2_det1 = np.sqrt(1.0/qfx2_det1)
        qfx2_K_det1 = np.rollaxis(np.tile(qfx2_det1, (K, 1)), 1)
        qfx2_det2 = np.array([cx2_kpts[cx][fx, [2,4]] for (cx, fx) in
                            izip(qfx2_cx.flat, qfx2_fx.flat)], np.float).prod(1)
        qfx2_det2.shape = (nQuery, K)
        qfx2_det2 = np.sqrt(1.0/qfx2_det2)
        qfx2_scaledist = qfx2_det2 / qfx2_K_det1
        cx2_weight[qcx] = qfx2_scaledist
    return cx2_weight


def nn_placketluce_score(hs, qcx2_neighbors, data_index, nn_params):
    pass

def nn_positional_score(hs, qcx2_neighbors, data_index, nn_params):
    pass
