import params
import helpers
import numpy as np
import pyflann
import algos
from itertools import izip, chain
def reload_module():
    import imp, sys
    print('reloading '+__name__)
    imp.reload(sys.modules[__name__])

def rrr():
    reload_module()

class NNIndex(object):
    def __init__(self, hs, sx2_cx):
        cx2_desc  = hs.feats.cx2_desc
        # Make unique id for indexed descriptors
        feat_uid   = params.get_feat_uid()
        sample_uid = helpers.make_sample_id(sx2_cx)
        uid = '_cxs(' + sample_uid + ')' + feat_uid
        # Number of features per sample chip
        sx2_nFeat = [len(cx2_desc[sx]) for sx in iter(sx2_cx)]
        # Inverted index from indexed descriptor to chipx and featx 
        _ax2_cx = [[cx]*nFeat for (cx, nFeat) in izip(sx2_cx, sx2_nFeat)]
        _ax2_fx = [range(nFeat) for nFeat in iter(sx2_nFeat)]
        ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
        ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
        # Aggregate indexed descriptors into continuous structure
        ax2_desc = np.vstack([cx2_desc[cx] for cx in sx2_cx])
        # Build/Load the flann index
        flann_params = {'algorithm':'kdtree', 'trees':4}
        precomp_kwargs = {'cache_dir'    : hs.dirs.cache_dir,
                          'uid'          : uid,
                          'flann_params' : flann_params, }
        flann = algos.precompute_flann(ax2_desc, **precomp_kwargs)
        #----
        # Agg Data
        self.ax2_cx   = ax2_cx
        self.ax2_fx   = ax2_fx
        self.ax2_data = ax2_desc
        self.flann = flann
    def __del__(self):
        if not self.flann:
            self.flann.delete_index()

class NNParams(DynStruct):
    def __init__(nn_params, **kwargs):
        super(NNParams, nn_params).__init__
        # Core
        nn_params.K = 2
        nn_params.Knorm = 1
        # Filters
        nn_params.nnfilter_list = ['reciprocal', 'roidist']
        #['reciprocal', 'roidist', 'frexquency', 'ratiotest', 'bursty']
        nn_params.K_reciprocal   = 1 # 0 := off
        nn_params.roidist_thresh = 1 # 1 := off
        nn_params.ratio_thresh   = 1 # 1 := off
        nn_params.freq_thresh    = 1 # 1 := off
        nn_params.checks = 128
        nn_params.__dict__.update(**kwargs)

class SpatialVerifyParams(DynStruct):
    def __init__(sv_params, **kwargs):
        super(SpatialVerifyParams, sv_params).__init__
        sv_params.scale_thresh  = (.5, 2)
        sv_params.xy_thresh = .002
        sv_params.shortlist_len = 100
        sv_params.__dict__.update(kwargs)

class ScoringParams(DynStruct):
    def __init__(score_params, **kwargs):
        super(ScoringParams, score_params).__init__
        score_params.aggregation_method = 'ChipSum' # ['NameSum', 'NamePlacketLuce']
        score_params.meta_params = {
            'roidist'    : (.5),
            'reciprocal' : (0), 
            'ratio'      : (1.2), 
            'scale'      : (.5),
            'bursty'     : (1),
            'lnbnn'      : 0, 
        }
        score_params.num_shortlist  = 100
        score_params.__dict__.update(kwargs)

class QueryParams(DynStruct):
    def __init__(query_params, **kwargs):
        super(QueryParams, query_params).__init__
        query_params.nn_params = NNParams(**kwargs)
        query_params.score_params = ScoringParams(**kwargs)
        query_params.sv_params = SpatialVerifyParams( *kwargs)
        query_params.query_type = 'vsmany'
        query_params.__dict__.update(kwargs)
