from hscom import helpers
from hscom import helpers as util
from hotspotter import nn_filters
from dbgimport import  *
'''
%run main.py --query 28 --nocache-query
'''


exec(open(nn_filters.__file__).read())
qcx2_nns = helpers.load_testdata('qcx2_nns')
qreq = hs.qreq
qcx = list(qcx2_nns.keys())[0]
K, Knorm, rule = qreq.cfg.nn_cfg.dynget('K', 'Knorm', 'normalizer_rule')
dx2_cx = qreq._data_index.ax2_cx
dx2_fx = qreq._data_index.ax2_fx

(qfx2_dx, qfx2_dist) = qcx2_nns[qcx]
qfx2_nndist = qfx2_dist[:, 0:K]
# Get the top names you do not want your normalizer to be from
qtnx = hs.cx2_tnx(qcx)
nTop = max(1, K)
qfx2_topdx = qfx2_dx.T[0:nTop, :].T
qfx2_normdx = qfx2_dx.T[-Knorm:].T
# Apply temporary uniquish name
qfx2_topcx = dx2_cx[qfx2_topdx]
qfx2_toptnx = hs.cx2_tnx(qfx2_topcx)
qfx2_normtcx = dx2_cx[qfx2_normdx]
qfx2_normtnx = hs.cx2_tnx(qfx2_normtcx)
# Inspect the potential normalizers
qfx2_normk = mark_name_valid_normalizers(qfx2_normtnx, qfx2_toptnx, qtnx)
qfx2_normk += (K + Knorm)  # convert form negative to pos indexes

qfx2_normdx = [dxs[normk] for (dxs, normk) in zip(qfx2_dx, qfx2_normk)]

qfx2_normcx = dx2_cx[qfx2_normdx]
qfx2_normnx = hs.cx2_tnx(qfx2_normcx)



qcx2_nns = [
    (1, 2, 3, 4, 5, 6, 7),
    (8, 9, 10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19, 20, 21),
    (22, 23, 24, 24, 26, 27, 28),
]
