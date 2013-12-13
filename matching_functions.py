from __future__ import division, print_function
import __builtin__
import sys
from itertools import izip
# Hotspotter
import DataStructures as ds
import helpers
import nn_filters
import spatial_verification2 as sv2
import voting_rules2 as vr2
# Scientific 
import numpy as np

MARK_AFTER = 40

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp
    print('[mf] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

def printDBG(msg):
    pass
    #print('[DBG] ' + msg)
    pass

#============================
# Nearest Neighbors
#============================
def nearest_neighbors(hs, qcxs, q_cfg):
    'Plain Nearest Neighbors'
    data_index = q_cfg.data_index
    flann = data_index.flann
    nn_cfg  = q_cfg.nn_cfg
    print(''.join(['[mf] Step 1) Assign nearest neighbors: ']+nn_cfg.get_uid()))
    K = nn_cfg.K
    Knorm = nn_cfg.Knorm
    checks = nn_cfg.checks
    cx2_desc = hs.feats.cx2_desc
    def nnfunc(qfx2_desc):
        #print(qfx2_desc.shape)
        #print(data_index)
        return flann.nn_index(qfx2_desc, K+Knorm, checks=checks)
    #qcx2_nns = {qcx:func(qcx) for qcx in qcxs}
    qcx2_nns = {}
    nNN = 0
    nDesc = 0
    def mark_progress(): pass
    if len(qcxs) > MARK_AFTER:
        def mark_progress(): print_('.')
    for qcx in qcxs:
        printDBG('[mf] Finding nearest neighbors of qcx=%r' % (qcx,))
        mark_progress()
        qfx2_desc = cx2_desc[qcx]
        (qfx2_dx, qfx2_dist) = nnfunc(qfx2_desc)
        qcx2_nns[qcx] = (qfx2_dx, qfx2_dist)
        nNN += qfx2_dx.size
        nDesc += len(qfx2_desc)
    if len(qcxs) > MARK_AFTER:
        print('')
    print('[mf] * assigned %d desc from %d chips to %r nearest neighbors' %
          (nDesc, len(qcxs), nNN))
    return qcx2_nns

#============================
# Nearest Neighbor weights
#============================
def weight_neighbors(hs, qcx2_nns, q_cfg):
    f_cfg = q_cfg.f_cfg
    if not f_cfg.filt_on:
        return  {}
    print(''.join(['[mf] Step 2) Weight neighbors: ']+f_cfg.get_uid()))
    nnfilter_list = f_cfg.nnfilter_list
    filt2_weights = {}
    for nnfilter in nnfilter_list:
        print('[mf] * computing %s weights' % nnfilter)
        nnfilter_fn = eval('nn_filters.nn_'+nnfilter+'_weight')
        filt2_weights[nnfilter] = nnfilter_fn(hs, qcx2_nns, q_cfg)
    return filt2_weights

#==========================
# Neighbor scoring (Voting Profiles)
#==========================
def _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw):
    qfx2_score = np.ones(qfx2_nn.shape, dtype=ds.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, cx2_weights in filt2_weights.iteritems():
        qfx2_weights = cx2_weights[qcx]
        (sign, thresh), weight = filt2_tw[filt]
        printDBG('[mf] * filt=%r ' % filt)
        if not thresh is None or not weight == 0:
            printDBG('[mf] * \\ qfx2_weights = %r' % helpers.printable_mystats(qfx2_weights.flatten()))
        if not thresh is None:
            qfx2_passed = sign*qfx2_weights <= sign*thresh
            nValid  = qfx2_valid.sum()
            qfx2_valid  = np.bitwise_and(qfx2_valid, qfx2_passed)
            nPassed = (True - qfx2_passed).sum()
            nAdded = nValid - qfx2_valid.sum()
            #print(str(sign*qfx2_weights))
            printDBG('[mf] * \\ *thresh=%r, nFailed=%r, nFiltered=%r' % (sign*thresh, nPassed, nAdded))
        if not weight == 0:
            printDBG('[mf] * \\ weight=%r' % weight)
            qfx2_score  += weight * qfx2_weights
    return qfx2_score, qfx2_valid

def filter_neighbors(hs, qcx2_nns, filt2_weights, q_cfg):
    qcx2_nnfilter = {}
    f_cfg = q_cfg.f_cfg
    data_index = q_cfg.data_index
    K = q_cfg.nn_cfg.K
    dx2_cx = data_index.ax2_cx
    printDBG('[mf] unique(dx2_cx) = %r ' % (np.unique(dx2_cx),))
    filt2_tw = f_cfg.filt2_tw
    print('[mf] Step 3) Filter neighbors')
    def mark_progress(): pass
    if len(qcx2_nns) > MARK_AFTER:
        def mark_progress(): print_('.')
    for qcx in qcx2_nns.iterkeys():
        mark_progress()
        printDBG('[mf] --------------')
        printDBG('[mf] * scoring q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_score, qfx2_valid = _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw)
        qfx2_cx = dx2_cx[qfx2_nn]
        # dont vote for yourself
        qfx2_notself_vote = qfx2_cx != qcx
        printDBG('[mf] * qcx  = %r ' % (qcx,))
        printDBG('[mf] * unique(qfx2_cx) = %r ' % (np.unique(qfx2_cx),))
        printDBG('[mf] * Removed %d/%d self-votes' % ((True - qfx2_notself_vote).sum(), qfx2_notself_vote.size))
        printDBG('[mf] * %d/%d valid neighbors ' % (qfx2_valid.sum(), qfx2_valid.size))
        qfx2_valid = np.bitwise_and(qfx2_valid, qfx2_notself_vote) 
        qcx2_nnfilter[qcx] = (qfx2_score, qfx2_valid)
    if len(qcx2_nns) > MARK_AFTER:
        print('')
    return qcx2_nnfilter

#-----
# Scoring Mechanism
#-----
#s2coring_func  = [LNBNN, PlacketLuce, TopK, Borda]
#load_precomputed(cx, q_cfg)
def score_chipmatch(hs, qcx, chipmatch, score_method, q_cfg=None):
    (cx2_fm, cx2_fs, cx2_fk) = chipmatch
    print('[mf] * Scoring chipmatch: %s cx=%r' % (score_method, qcx))
    if score_method == 'csum':
        cx2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'nsum':
        cx2_score, nx2_score = score_chipmatch_nsum(hs, qcx, chipmatch, q_cfg)
    elif score_method == 'nunique':
        cx2_score, nx2_score = score_chipmatch_nunique(hs, qcx, chipmatch, q_cfg)
    elif score_method == 'pl':
        cx2_score, nx2_score = vr2.score_chipmatch_PL(hs, qcx, chipmatch, q_cfg)
    elif score_method == 'borda':
        cx2_score, nx2_score = vr2.score_chipmatch_pos(hs, qcx, chipmatch, q_cfg, 'borda')
    elif score_method == 'topk':
        cx2_score, nx2_score = vr2.score_chipmatch_pos(hs, qcx, chipmatch, q_cfg, 'topk')
    else:
        raise Exception('[mf] unknown scoring method:'+score_method)
    cx2_nMatch = np.array(map(len, cx2_fm))
    # Autoremove chips with no score
    cx2_score = (cx2_score * (cx2_nMatch != 0))
    # Autoremove chips which are not the top scoring in their name
    cx2_score = vr2.enforce_one_name_per_cscore(hs, cx2_score, chipmatch)
    return cx2_score

#============================
# Conversion qfx2 -> cx2
#============================
def build_chipmatches(hs, qcx2_nns, qcx2_nnfilt, q_cfg):
    '''vsmany/vsone counts here. also this is where the filter 
    weights and thershold are applied to the matches. Essientally 
    nearest neighbors are converted into weighted assignments'''
    print('[mf] Step 4) Building chipmatches')
    data_index = q_cfg.data_index
    query_type = q_cfg.a_cfg.query_type
    K = q_cfg.nn_cfg.K
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs   = q_cfg.dcxs 
    invert_query = query_type == 'vsone'
    qcx2_chipmatch = {}
    #Vsone
    if invert_query: 
        assert len(q_cfg.qcxs) == 1
        cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)
    # Iterate over chips with nearest neighbors
    def mark_progress(): pass
    if len(qcx2_nns) > MARK_AFTER:
        def mark_progress(): print_('.')
    for qcx in qcx2_nns.iterkeys():
        mark_progress()
        #print('[mf] * scoring q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        (qfx2_fs, qfx2_valid) = qcx2_nnfilt[qcx]
        nQuery = len(qfx2_dx)
        # Build feature matches
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_qfx = np.tile(np.arange(nQuery), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQuery, 1))
        qfx2_tup = (qfx2_qfx, qfx2_cx, qfx2_fx, qfx2_fs, qfx2_k)
        match_iter = izip(*[qfx2[qfx2_valid] for qfx2 in qfx2_tup])
        # Vsmany
        if not invert_query: 
            cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)
            for qfx, cx, fx, fs, fk in match_iter:
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(fs)
                cx2_fk[cx].append(fk)
            chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
            qcx2_chipmatch[qcx] = chipmatch
            continue
        # Vsone
        for qfx, cx, fx, fs, fk in match_iter:
            cx2_fm[qcx].append((fx, qfx))
            cx2_fs[qcx].append(fs)
            cx2_fk[qcx].append(fk)
    #Vsone
    if invert_query:
        chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
        qcx = q_cfg.qcxs[0]
        qcx2_chipmatch[qcx] = chipmatch
    if len(qcx2_nns.keys()) > MARK_AFTER:
        print('')
    return qcx2_chipmatch

#============================
# Conversion to cx2 -> qfx2
#============================
def chipmatch2_neighbors(hs, qcx2_chipmatch, q_cfg):
    raise NotImplemented('almost')
    qcx2_nns={}
    K = q_cfg.nn_cfg.K
    for qcx in qcx2_chipmatch.iterkeys():
        nQuery = len(hs.feats.cx2_kpts[qcx])
        # Stack the feature matches
        (cx2_fm, cx2_fs, cx2_fk) = qcx2_chipmatch[qcx]
        cxs = np.hstack([[cx]*len(cx2_fm[cx]) for cx in xrange(len(cx2_fm))])
        fms = np.vstack(cx2_fm)
        # Get the individual feature match lists
        qfxs = fms[:,0]
        fxs  = fms[:,0]
        fss  = np.hstack(cx2_fs)
        fks  = np.hstack(cx2_fk)
        # Rebuild the nearest neigbhor matrixes
        qfx2_cx = -np.ones((nQuery, K), np.int)
        qfx2_fx = -np.ones((nQuery, K), np.int)
        qfx2_fs = -np.ones((nQuery, K), ds.FS_DTYPE)
        qfx2_valid = np.zeros((nQuery, K), np.bool)
        # Populate nearest neigbhor matrixes
        for qfx, k in izip(qfxs, fks):
            assert qfx2_valid[qfx, k] == False
            qfx2_valid[qfx, k] = True
        for cx, qfx, k in izip(cxs, qfxs, fks): qfx2_cx[qfx, k] = cx
        for qfx, fx, k in izip(qfxs, fxs, fks): qfx2_fx[qfx, k] = fx
        for qfx, fs, k in izip(qfxs, fss, fks): qfx2_fs[qfx, k] = fs
        nns = (qfx2_cx, qfx2_fx, qfx2_fs, qfx2_valid)
        qcx2_nns[qcx] = nns
    return qcx2_nns

#-----
# Spatial Verification
#-----

def _precompute_topx2_dlen_sqrd(cx2_rchip_size, cx2_kpts, cx2_fm, topx2_cx,
                                nRerank, use_chip_extent, USE_1_to_2):
    '''helper for spatial verification, computes the squared diagonal length of
    matching chips'''
    if use_chip_extent:
        def cx2_chip_dlensqrd(cx):
            (chipw, chiph) = cx2_rchip_size[cx]
            dlen_sqrd = chipw**2 + chiph**2
            return dlen_sqrd
        if USE_1_to_2:
            topx2_dlen_sqrd = [cx2_chip_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
        else:
            topx2_dlen_sqrd = [cx2_chip_dlensqrd(qcx)]*nRerank
    else:
        if USE_1_to_2:
            def cx2_kpts2_dlensqrd(cx):
                kpts2 = cx2_kpts[cx]
                fm    = cx2_fm[cx]
                if len(fm) == 0:
                    return 1
                x_m = kpts2[fm[:,1],0].T
                y_m = kpts2[fm[:,1],1].T
                return (x_m.max()-x_m.min())**2 + (y_m.max()-y_m.min())**2
            topx2_dlen_sqrd = [cx2_kpts2_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
        else:
            def cx2_kpts1_dlensqrd(cx):
                kpts2 = cx2_kpts[cx]
                fm    = cx2_fm[cx]
                if len(fm) == 0:
                    return 1
                x_m = kpts2[fm[:,0],0].T
                y_m = kpts2[fm[:,0],1].T
                return (x_m.max()-x_m.min())**2 + (y_m.max()-y_m.min())**2
            topx2_dlen_sqrd = [cx2_kpts1_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
    return topx2_dlen_sqrd
#if USE_2_to_1:
    #sv_tup = sv2.homography_inliers(kpts2, kpts1, np.hstack(fm[:,1], fm[:,0]), xy_thresh, max_scale,
                                    #min_scale, dlen_sqrd, min_nInliers,
                                #just_affine)
#else
#if not USE_2_to_1:
#if not use_chip_extent or USE_1_to_2:

def spatial_verification(hs, qcx2_chipmatch, q_cfg):
    sv_cfg = q_cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        return qcx2_chipmatch
    print('[mf] Step 5) Spatial verification: %r' % ''.join(sv_cfg.get_uid()))
    prescore_method  = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    min_scale, max_scale = sv_cfg.scale_thresh
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    just_affine     = sv_cfg.just_affine
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts = hs.feats.cx2_kpts
    qcx2_chipmatchSV = {}
    USE_1_to_2 = True
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qcx in qcx2_chipmatch.iterkeys():
        #print('[mf] verify qcx=%r' % qcx)
        chipmatch = qcx2_chipmatch[qcx]
        cx2_prescore = score_chipmatch(hs, qcx, chipmatch, prescore_method, q_cfg)
        (cx2_fm, cx2_fs, cx2_fk) = chipmatch
        topx2_cx= cx2_prescore.argsort()[::-1]
        nRerank = min(len(topx2_cx), nShortlist)
        # Precompute output container
        cx2_fm_V, cx2_fs_V, cx2_fk_V = new_fmfsfk(hs)
        # Query Keypoints
        kpts1 = cx2_kpts[qcx]
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = _precompute_topx2_dlen_sqrd(cx2_rchip_size, cx2_kpts,
                                                      cx2_fm, topx2_cx, nRerank,
                                                      use_chip_extent,
                                                      USE_1_to_2)
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            #print('[mf] vs topcx=%r, score=%r' % (cx, cx2_prescore[cx]))
            cx = topx2_cx[topx]
            fm = cx2_fm[cx]
            if len(fm) < min_nInliers:
                print_('x')
                continue
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = cx2_kpts[cx]
            fs    = cx2_fs[cx]
            fk    = cx2_fk[cx]
            sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, max_scale,
                                            min_scale, dlen_sqrd, min_nInliers,
                                        just_affine)
            if sv_tup is None:
                print_('o')
                continue
            # Return the inliers to the homography
            #print('fm=%r' % (fm,))
            #print('fs=%r' % (fs,))
            #print('fk=%r' % (fk,))
            (H, inliers) = sv_tup
            cx2_fm_V[cx] = fm[inliers, :]
            cx2_fs_V[cx] = fs[inliers]
            cx2_fk_V[cx] = fk[inliers]
            print_('.')
            #np.set_printoptions(threshold=2)
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(cx2_fm_V, cx2_fs_V, cx2_fk_V)
        qcx2_chipmatchSV[qcx] = chipmatchSV
    print('\n[mf] Finished sv')
    return qcx2_chipmatchSV
#-----

def _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk):
    # Convert to numpy
    fm_dtype_ = ds.FM_DTYPE
    fs_dtype_ = ds.FS_DTYPE
    fk_dtype_ = ds.FK_DTYPE
    cx2_fm = [np.array(fm, fm_dtype_) for fm in iter(cx2_fm)]
    cx2_fs = [np.array(fs, fs_dtype_) for fs in iter(cx2_fs)]
    cx2_fk = [np.array(fk, fk_dtype_) for fk in iter(cx2_fk)]
    # Ensure shape
    for cx in xrange(len(cx2_fm)):
        cx2_fm[cx].shape = (cx2_fm[cx].size//2, 2)
    # Cast lists
    cx2_fm = np.array(cx2_fm, list)
    cx2_fs = np.array(cx2_fs, list)
    cx2_fk = np.array(cx2_fk, list)
    chipmatch = (cx2_fm, cx2_fs, cx2_fk)
    return chipmatch

def new_fmfsfk(hs):
    cx2_fm = [[] for _ in xrange(hs.num_cx)]
    cx2_fs = [[] for _ in xrange(hs.num_cx)]
    cx2_fk = [[] for _ in xrange(hs.num_cx)]
    return cx2_fm, cx2_fs, cx2_fk

#----------
# QueryResult Format
#----------

def special_uids(q_cfg, aug):
    real_uid = q_cfg.get_uid() + aug
    # Hacky dev stuff
    if aug == '+NN':
        title_uid = q_cfg.get_uid(  'NN', 'noFILT', 'noSV', 'noAGG', 'noCHIP')
    elif aug == '+FILT':
        title_uid = q_cfg.get_uid('noNN',   'FILT', 'noSV', 'noAGG', 'noCHIP')
    elif aug == '+SVER':
        title_uid = q_cfg.get_uid('noNN', 'noFILT',   'SV', 'noAGG', 'noCHIP')
    else:
        title_uid = q_cfg.get_uid()
    return real_uid, title_uid

def load_resdict(hs, qcxs, q_cfg, aug=''):
    real_uid, title_uid = special_uids(q_cfg, aug)
    # Load the result structures for each query.
    try: 
        qcx2_res = {}
        for qcx in qcxs:
            res = ds.QueryResult(qcx, real_uid, q_cfg)
            res.load(hs)
            qcx2_res[qcx] = res
    except IOError as ex:
        return None
    return qcx2_res

# qcx2_chipmatch = matchesSVER
def chipmatch_to_resdict(hs, qcx2_chipmatch, q_cfg, aug=''):
    print('[mf] Step 6) Convert chipmatch -> res')
    real_uid, title_uid = special_uids(q_cfg, aug)
    score_method = q_cfg.a_cfg.score_method
    # Create the result structures for each query.
    qcx2_res = {}
    for qcx in qcx2_chipmatch.iterkeys():
        chipmatch = qcx2_chipmatch[qcx]
        cx2_score = score_chipmatch(hs, qcx, chipmatch, score_method, q_cfg)
        res = ds.QueryResult(qcx, real_uid, q_cfg)
        res.cx2_score = cx2_score
        (res.cx2_fm, res.cx2_fs, res.cx2_fk) = chipmatch
        res.title = (title_uid + ' ' + aug).strip(' ')
        qcx2_res[qcx] = res
    # Retain original score method
    return qcx2_res
