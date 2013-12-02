FeatureConfig

    
def load_chip_features2(hs, load_kpts=True, load_desc=True):
    print('\n=============================')
    print('[fc2] Computing and loading features for %r' % hs.db_name())
    print('=============================')
    return load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)


def load_features(hs, cx_list=None, fc_cfg=None, **kwargs):
    if fc_cfg is None: 
        fc_cfg = FeatureConfig(**kwargs)
    feat_dict = {}
    feat_dir       = hs.dirs.feat_dir
    cx2_rchip_path = hs.cpaths.cx2_rchip_path
    cx2_cid        = hs.tables.cx2_cid
    valid_cxs = hs.get_valid_cxs()

    def precompute_feat_type(hs, feat_type, cx_list):
        if not feat_type in feat_dict.keys():
            feat_dict[feat_type] = Features(feat_type)
        feat = feat_dict[feat_type]
        # Build Parallel Jobs, Compute features, saving them to disk.
        # Then Run Parallel Jobs 
        cid_iter = (cx2_cid[cx] for cx in cx_list)
        feat_type_str = helpers.remove_chars(repr(feat_type), [' ', '(', ')', '\''])
        cx2_feat_path = [feat_dir+'/CID_%d_%s.npz' % (cid, feat_type_str) for cid in cid_iter]
        precompute_fn = feat_type2_precompute[feat_type]
        parallel_compute(precompute_fn, [cx2_rchip_path, cx2_feat_path])

    for feat_type in feature_types:
        precompute_feat_type(hs, feat_type, cx_list)



# Decorators? 
#def cache_features():
#def uncache_features():

# hs is a handle to whatever data is loaded 
# if you aren't using something
# del hs.something
# EG: del hs.feats
'''
 del.hs.feats['sift']
 ax2_features = hs.feats['sift']
 ax2_features = hs.feats['constructed-sift']
 ax2_features = hs.feats['mser']
'''

def index_features(hs):
    mser_feats = hs.feats['mser']
    sift_feats = hs.feats['sift']

    feats = sift_feats
    ax2_desc = feats.ax2_desc
    ax2_kpts = feats.ax2_kpts
    cx2_nFeats = feats.cx2_nFeats
    cx2_axStart = np.cumsum(cx2_nFeats)
    def cx2_axs(cx):
        nFeats  = cx2_nFeats[cx]
        axStart = cx2_axStart[cx]
        fx2_ax = np.arange(nFeats, axStart+nFeats)
        return fx2_ax

    hs.feats.cx2_axs = cx2_axs

    #ax_in_grid[0,0] = [...]
    #ax_in_part[0]   = [...]
    #valid_ax        = [...]

def test2(hs):
    cx_list = hs.get_valid_cxs()
    #feature_types = params.FEAT_TYPE
    feature_types = [('hesaff','sift'), ('mser','sift')]
    feat_type = 'mser'
    load_features(hs, cx_list, feature_types)


    
def load_chip_features2(hs, load_kpts=True, load_desc=True):
    print('\n=============================')
    print('[fc2] Computing and loading features for %r' % hs.db_name())
    print('=============================')
    return load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)
