'''
Draws all chips belonging to a name and dumps them into the 
[results]/groundtruth directory
'''
from __init__ import *
from __future__ import division
import matplotlib.gridspec as gridspec
import vizualizations as viz
import draw_func2 as df2
import numpy as np
import params
import load_data2 as ld2

def reload_module():
    import imp
    import sys
    print('[df2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def dump_all_groundtruths(hs):
    print('Dumping all groundtruth')
    nx2_name = hs.tables.nx2_name
    nx2_cxs = hs.get_nx2_cxs()
    nx2_ncxs = np.array(map(len, nx2_cxs))
    nx_sort = nx2_ncxs.argsort()
    nx_sort = nx_sort[::-1]
    print('num names = %r' % len(nx_sort))
    print('max/min = %r / %r' % (nx2_ncxs.max(), nx2_ncxs.min()))
    for nx in iter(nx_sort):
        if nx2_ncxs[nx] == 0:
            continue
        if nx <= 1:
            continue
        plot_name(hs, nx, nx2_cxs)
        viz.__dump(hs, 'groundtruth')

def plot_cx(hs, cx):
    ax = df2.plt.gca()
    rchip = hs.get_chip(cx)
    ax.imshow(rchip, interpolation='nearest')
    df2.plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])
    gname = hs.cx2_gname(cx)
    cid = hs.tables.cx2_cid[cx]
    ax.set_xlabel(gname)
    ax.set_title(hs.cxstr(cx))

def plot_name_cx(hs, cx, **kwargs):
    nx = hs.tables.cx2_nx[cx]
    plot_name(hs, nx, **kwargs)

def plot_name(hs, nx, nx2_cxs=None, fignum=0, **kwargs):
    print('[viz*] plot_name nx=%r' % nx)
    if not 'fignum' in vars():
        kwargs = {}
        fignum = 0
    nx2_name = hs.tables.nx2_name
    cx2_nx   = hs.tables.cx2_nx
    name = nx2_name[nx]
    if not nx2_cxs is None:
        cxs = nx2_cxs[nx]
    else: 
        cxs = np.where(cx2_nx == nx)[0]
    print('[viz*] plot_name %r' % hs.cxstr(cxs))
    ncxs  = len(cxs)
    nCols = int(min(np.ceil(np.sqrt(ncxs)), 5))
    nRows = int(np.ceil(ncxs / nCols))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    gs2 = gridspec.GridSpec(nRows, nCols)
    fig = df2.figure(fignum=fignum, **kwargs)
    fig.clf()
    for ss, cx in zip(gs2, cxs):
        ax = fig.add_subplot(ss)
        plot_cx(hs, cx)
    title = 'nx=%r -- name=%r' % (nx, name)
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    df2.set_figtitle(title)


if __name__ == '__main__':
    #params.DEFAULT = params.TOADS
    db_dir = params.DEFAULT
    hs = ld2.HotSpotter()
    hs.load_basic(db_dir)
    dump_all_groundtruths(hs)
