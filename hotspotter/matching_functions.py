from __future__ import division, print_function
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, '[mf]', DEBUG=False)
# Python
from itertools import izip
# Scientific
import numpy as np
# Hotspotter
import QueryResult as qr
import nn_filters
import spatial_verification2 as sv2
import voting_rules2 as vr2
from hscom import helpers


#=================
# Module Concepts
#=================
'''
PREFIXES:
qcx2_XXX - prefix mapping query chip index to
qfx2_XXX  - prefix mapping query chip feature index to

TUPLES:
 * nns    - a (qfx2_dx, qfx2_dist) tuple
 * nnfilt - a (qfx2_fs, qfx2_valid) tuple

SCALARS
 * dx     - the index into the database of features
 * dist   - the distance to a corresponding feature
 * fs     - a score of a corresponding feature
 * valid  - a valid bit for a corresponding feature

REALIZATIONS:
qcx2_nns - maping from query chip index to nns
{
  * qfx2_dx    - a ranked list of query feature indexes to database feature indexes
  * qfx2_dist  - a ranked list of query feature indexes to database feature indexes
}

* qcx2_norm_weight - mapping from qcx to (qfx2_normweight, qfx2_selnorm)

         = qcx2_nnfilt[qcx]


'''
#=================
# Globals
#=================

# TODO: Make a more elegant way of mapping weighting parameters to weighting
# function. A dict is better than eval, but there may be a better way.
NN_FILTER_FUNC_DICT = {
    'scale':   nn_filters.nn_scale_weight,
    'roidist': nn_filters.nn_roidist_weight,
    'recip':   nn_filters.nn_recip_weight,
    'bursty':  nn_filters.nn_bursty_weight,
    'lnrat':   nn_filters.nn_lnrat_weight,
    'lnbnn':   nn_filters.nn_lnbnn_weight,
    'ratio':   nn_filters.nn_ratio_weight,
}
MARK_AFTER = 40

#=================
# Helpers
#=================


def progress_func(maxval=0):
    mark_progress, end_progress = helpers.progress_func(maxval, mark_after=MARK_AFTER, progress_type='simple')
    #if maxval > MARK_AFTER:
        #print('')
    return mark_progress, end_progress


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(hs, qcx):
    ex = QueryException('Query %r has no descriptors! Please delete it.' % hs.cidstr(qcx))
    return ex


#============================
# 1) Nearest Neighbors
#============================


@profile
def nearest_neighbors(hs, qcxs, qdat):
    'Plain Nearest Neighbors'
    # Neareset neighbor configuration
    nn_cfg = qdat.cfg.nn_cfg
    K      = nn_cfg.K
    Knorm  = nn_cfg.Knorm
    checks = nn_cfg.checks
    uid_   = nn_cfg.get_uid()
    print('[mf] Step 1) Assign nearest neighbors: ' + uid_)
    # Grab descriptors
    cx2_desc = hs.feats.cx2_desc
    # NNIndex
    flann = qdat._data_index.flann
    # Output
    qcx2_nns = {}
    nNN, nDesc = 0, 0
    mark_progress, end_progress = progress_func(len(qcxs))
    for count, qcx in enumerate(qcxs):
        mark_progress(count)
        qfx2_desc = cx2_desc[qcx]
        # Check that we can query this chip
        if len(qfx2_desc) == 0:
            raise NoDescriptorsException(hs, qcx)
        # Find Neareset Neighbors
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        # Store nearest neighbors
        qcx2_nns[qcx] = (qfx2_dx, qfx2_dist)
        # record number of query and result desc
        nNN += qfx2_dx.size
        nDesc += len(qfx2_desc)
    end_progress()
    print('[mf] * assigned %d desc from %d chips to %r nearest neighbors' %
          (nDesc, len(qcxs), nNN))
    return qcx2_nns


#============================
# 2) Nearest Neighbor weights
#============================


@profile
def weight_neighbors(hs, qcx2_nns, qdat):
    filt_cfg = qdat.cfg.filt_cfg
    print('[mf] Step 2) Weight neighbors: ' + filt_cfg.get_uid())
    if not filt_cfg.filt_on:
        return  {}
    nnfilter_list = filt_cfg._nnfilter_list
    filt2_weights = {}
    filt2_meta = {}
    for nnfilter in nnfilter_list:
        nn_filter_fn = NN_FILTER_FUNC_DICT[nnfilter]
        # Apply [nnfilter] weight to each nearest neighbor
        # TODO FIX THIS!
        qcx2_norm_weight, qcx2_selnorms = nn_filter_fn(hs, qcx2_nns, qdat)
        filt2_weights[nnfilter] = qcx2_norm_weight
        filt2_meta[nnfilter] = qcx2_selnorms
    return filt2_weights, filt2_meta


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


@profile
def filter_neighbors(hs, qcx2_nns, filt2_weights, qdat):
    qcx2_nnfilter = {}
    # Configs
    filt_cfg = qdat.cfg.filt_cfg
    cant_match_sameimg = not filt_cfg.can_match_sameimg
    cant_match_samename = not filt_cfg.can_match_samename
    K = qdat.cfg.nn_cfg.K
    filt2_tw = filt_cfg._filt2_tw
    print('[mf] Step 3) Filter neighbors: ' + filt_cfg.get_uid())
    # NNIndex
    # Database feature index to chip index
    dx2_cx = qdat._data_index.ax2_cx
    # Filter matches based on config and weights
    mark_progress, end_progress = progress_func(len(qcx2_nns))
    for count, qcx in enumerate(qcx2_nns.iterkeys()):
        mark_progress(count)
        (qfx2_dx, _) = qcx2_nns[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw)
        qfx2_cx = dx2_cx[qfx2_nn]
        print('[mf] * %d assignments are invalid by thresh' % ((True - qfx2_valid).sum()))
        # Remove Impossible Votes:
        # dont vote for yourself or another chip in the same image
        qfx2_notsamechip = qfx2_cx != qcx
        cant_match_self = True
        if cant_match_self:
            ####DBG
            nChip_all_invalid = ((True - qfx2_notsamechip)).sum()
            nChip_new_invalid = (qfx2_valid * (True - qfx2_notsamechip)).sum()
            print('[mf] * %d assignments are invalid by self' % nChip_all_invalid)
            print('[mf] * %d are newly invalided by self' % nChip_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        if cant_match_sameimg:
            qfx2_notsameimg  = hs.tables.cx2_gx[qfx2_cx] != hs.tables.cx2_gx[qcx]
            ####DBG
            nImg_all_invalid = ((True - qfx2_notsameimg)).sum()
            nImg_new_invalid = (qfx2_valid * (True - qfx2_notsameimg)).sum()
            print('[mf] * %d assignments are invalid by gx' % nImg_all_invalid)
            print('[mf] * %d are newly invalided by gx' % nImg_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
        if cant_match_samename:
            qfx2_notsamename = hs.tables.cx2_nx[qfx2_cx] != hs.tables.cx2_nx[qcx]
            ####DBG
            nName_all_invalid = ((True - qfx2_notsamename)).sum()
            nName_new_invalid = (qfx2_valid * (True - qfx2_notsamename)).sum()
            print('[mf] * %d assignments are invalid by nx' % nName_all_invalid)
            print('[mf] * %d are newly invalided by nx' % nName_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
        print('[mf] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        qcx2_nnfilter[qcx] = (qfx2_score, qfx2_valid)
    end_progress()
    return qcx2_nnfilter


def _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw):
    qfx2_score = np.ones(qfx2_nn.shape, dtype=qr.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, cx2_weights in filt2_weights.iteritems():
        qfx2_weights = cx2_weights[qcx]
        (sign, thresh), weight = filt2_tw[filt]
        if isinstance(thresh, (int, float)):
            qfx2_passed = sign * qfx2_weights <= sign * thresh
            qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if not weight == 0:
            qfx2_score  += weight * qfx2_weights
    return qfx2_score, qfx2_valid


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> cx2
#============================


@profile
def build_chipmatches(hs, qcx2_nns, qcx2_nnfilt, qdat):
    '''vsmany/vsone counts here. also this is where the filter
    weights and thershold are applied to the matches. Essientally
    nearest neighbors are converted into weighted assignments'''
    print('[mf] Step 4) Building chipmatches')
    # Config
    query_type = qdat.cfg.agg_cfg.query_type
    K = qdat.cfg.nn_cfg.K
    is_vsone = query_type == 'vsone'
    # Data Index
    dx2_cx = qdat._data_index.ax2_cx
    dx2_fx = qdat._data_index.ax2_fx
    # Return var
    qcx2_chipmatch = {}

    #Vsone
    if is_vsone:
        assert len(qdat._qcxs) == 1
        cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)

    # Iterate over chips with nearest neighbors
    mark_progress, end_progress = progress_func(len(qcx2_nns))
    for count, qcx in enumerate(qcx2_nns.iterkeys()):
        mark_progress(count)
        #print('[mf] * scoring q' + hs.cidstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        (qfx2_fs, qfx2_valid) = qcx2_nnfilt[qcx]
        nQuery = len(qfx2_dx)
        # Build feature matches
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_qfx = np.tile(np.arange(nQuery), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQuery, 1))
        # Pack feature matches into an interator
        match_iter = izip(*[qfx2[qfx2_valid] for qfx2 in
                            (qfx2_qfx, qfx2_cx, qfx2_fx, qfx2_fs, qfx2_k)])
        if not is_vsone:
            cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)
            # Vsmany - Iterate over feature matches
            for qfx, cx, fx, fs, fk in match_iter:
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(fs)
                cx2_fk[cx].append(fk)
            chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
            qcx2_chipmatch[qcx] = chipmatch
        else:
            # Vsone - Iterate over feature matches
            for qfx, cx, fx, fs, fk in match_iter:
                cx2_fm[qcx].append((fx, qfx))  # Note the difference
                cx2_fs[qcx].append(fs)
                cx2_fk[qcx].append(fk)
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
        qcx = qdat._qcxs[0]
        qcx2_chipmatch[qcx] = chipmatch

    end_progress()
    return qcx2_chipmatch


#============================
# 5) Spatial Verification
#============================


@profile
def spatial_verification(hs, qcx2_chipmatch, qdat):
    sv_cfg = qdat.cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        print('[mf] Step 5) Spatial verification: off')
        return qcx2_chipmatch
    print('[mf] Step 5) Spatial verification: ' + sv_cfg.get_uid())
    prescore_method = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    min_scale = sv_cfg.scale_thresh_low
    max_scale = sv_cfg.scale_thresh_high
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    just_affine     = sv_cfg.just_affine
    cx2_rchip_size  = hs.cpaths.cx2_rchip_size
    cx2_kpts = hs.feats.cx2_kpts
    qcx2_chipmatchSV = {}
    #printDBG(qdat._dcxs)
    dcxs_ = set(qdat._dcxs)
    USE_1_to_2 = True
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qcx in qcx2_chipmatch.iterkeys():
        #printDBG('[mf] verify qcx=%r' % qcx)
        chipmatch = qcx2_chipmatch[qcx]
        cx2_prescore = score_chipmatch(hs, qcx, chipmatch, prescore_method, qdat)
        (cx2_fm, cx2_fs, cx2_fk) = chipmatch
        topx2_cx = cx2_prescore.argsort()[::-1]  # Only allow indexed cxs to be in the top results
        topx2_cx = [cx for cx in iter(topx2_cx) if cx in dcxs_]
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
            cx = topx2_cx[topx]
            fm = cx2_fm[cx]
            #printDBG('[mf] vs topcx=%r, score=%r' % (cx, cx2_prescore[cx]))
            #printDBG('[mf] len(fm)=%r' % (len(fm)))
            if len(fm) >= min_nInliers:
                dlen_sqrd = topx2_dlen_sqrd[topx]
                kpts2 = cx2_kpts[cx]
                fs    = cx2_fs[cx]
                fk    = cx2_fk[cx]
                #printDBG('[mf] computing homog')
                sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh,
                                                max_scale, min_scale, dlen_sqrd,
                                                min_nInliers, just_affine)
                #printDBG('[mf] sv_tup = %r' % (sv_tup,))
                if sv_tup is None:
                    print_('o')  # sv failure
                else:
                    # Return the inliers to the homography
                    (H, inliers) = sv_tup
                    cx2_fm_V[cx] = fm[inliers, :]
                    cx2_fs_V[cx] = fs[inliers]
                    cx2_fk_V[cx] = fk[inliers]
                    print_('.')  # verified something
            else:
                print_('x')  # not enough initial matches
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(cx2_fm_V, cx2_fs_V, cx2_fk_V)
        qcx2_chipmatchSV[qcx] = chipmatchSV
    print('')
    print('[mf] Finished sv')
    return qcx2_chipmatchSV


def _precompute_topx2_dlen_sqrd(cx2_rchip_size, cx2_kpts, cx2_fm, topx2_cx,
                                nRerank, use_chip_extent, USE_1_to_2):
    '''helper for spatial verification, computes the squared diagonal length of
    matching chips'''
    if use_chip_extent:
        def cx2_chip_dlensqrd(cx):
            (chipw, chiph) = cx2_rchip_size[cx]
            dlen_sqrd = chipw ** 2 + chiph ** 2
            return dlen_sqrd
        if USE_1_to_2:
            topx2_dlen_sqrd = [cx2_chip_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
        #else:
            #topx2_dlen_sqrd = [cx2_chip_dlensqrd(cx)] * nRerank
    else:
        if USE_1_to_2:
            def cx2_kpts2_dlensqrd(cx):
                kpts2 = cx2_kpts[cx]
                fm    = cx2_fm[cx]
                if len(fm) == 0:
                    return 1
                x_m = kpts2[fm[:, 1], 0].T
                y_m = kpts2[fm[:, 1], 1].T
                return (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            topx2_dlen_sqrd = [cx2_kpts2_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
        #else:
            #def cx2_kpts1_dlensqrd(cx):
                #kpts2 = cx2_kpts[cx]
                #fm    = cx2_fm[cx]
                #if len(fm) == 0:
                    #return 1
                #x_m = kpts2[fm[:, 0], 0].T
                #y_m = kpts2[fm[:, 0], 1].T
                #return (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            #topx2_dlen_sqrd = [cx2_kpts1_dlensqrd(cx) for cx in iter(topx2_cx[:nRerank])]
    return topx2_dlen_sqrd


def _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk):
    # Convert to numpy
    fm_dtype_ = qr.FM_DTYPE
    fs_dtype_ = qr.FS_DTYPE
    fk_dtype_ = qr.FK_DTYPE
    cx2_fm = [np.array(fm, fm_dtype_) for fm in iter(cx2_fm)]
    cx2_fs = [np.array(fs, fs_dtype_) for fs in iter(cx2_fs)]
    cx2_fk = [np.array(fk, fk_dtype_) for fk in iter(cx2_fk)]
    # Ensure shape
    for cx in xrange(len(cx2_fm)):
        cx2_fm[cx].shape = (cx2_fm[cx].size // 2, 2)
    # Cast lists
    cx2_fm = np.array(cx2_fm, list)
    cx2_fs = np.array(cx2_fs, list)
    cx2_fk = np.array(cx2_fk, list)
    chipmatch = (cx2_fm, cx2_fs, cx2_fk)
    return chipmatch


def new_fmfsfk(hs):
    num_chips = hs.get_num_chips()
    cx2_fm = [[] for _ in xrange(num_chips)]
    cx2_fs = [[] for _ in xrange(num_chips)]
    cx2_fk = [[] for _ in xrange(num_chips)]
    return cx2_fm, cx2_fs, cx2_fk


#============================
# 6) QueryResult Format
#============================


@profile
def chipmatch_to_resdict(hs, qcx2_chipmatch, filt2_meta, qdat, aug=''):
    print('[mf] Step 6) Convert chipmatch -> res')
    real_uid, title_uid = special_uids(qdat, aug)
    score_method = qdat.cfg.agg_cfg.score_method
    # Create the result structures for each query.
    qcx2_res = {}
    for qcx in qcx2_chipmatch.iterkeys():
        chipmatch = qcx2_chipmatch[qcx]
        cx2_score = score_chipmatch(hs, qcx, chipmatch, score_method, qdat)
        res = qr.QueryResult(qcx, real_uid, qdat)
        res.cx2_score = cx2_score
        (res.cx2_fm, res.cx2_fs, res.cx2_fk) = chipmatch
        res.title = (title_uid + ' ' + aug).strip(' ')
        res.filt2_meta = {}
        for filt, qcx2_meta in filt2_meta.iteritems():
            res.filt2_meta[filt] = qcx2_meta[qcx]
        qcx2_res[qcx] = res
    # Retain original score method
    return qcx2_res


def special_uids(qdat, aug):
    real_uid = qdat.cfg.get_uid() + aug
    # Hacky dev stuff
    if aug == '+NN':
        title_uid = qdat.cfg.get_uid(  'NN', 'noFILT', 'noSV', 'noAGG', 'noCHIP')
    elif aug == '+FILT':
        title_uid = qdat.cfg.get_uid('noNN',   'FILT', 'noSV', 'noAGG', 'noCHIP')
    elif aug == '+SVER':
        title_uid = qdat.cfg.get_uid('noNN', 'noFILT',   'SV', 'noAGG', 'noCHIP')
    else:
        title_uid = qdat.cfg.get_uid()
    return real_uid, title_uid


def load_resdict(hs, qcxs, qdat, aug=''):
    real_uid, title_uid = special_uids(qdat, aug)
    # Load the result structures for each query.
    try:
        qcx2_res = {}
        for qcx in qcxs:
            res = qr.QueryResult(qcx, real_uid, qdat)
            res.load(hs)
            qcx2_res[qcx] = res
    except IOError:
        return None
    return qcx2_res


#============================
# Scoring Mechanism
#============================


@profile
def score_chipmatch(hs, qcx, chipmatch, score_method, qdat=None):
    (cx2_fm, cx2_fs, cx2_fk) = chipmatch
    if score_method == 'csum':
        cx2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'pl':
        cx2_score, nx2_score = vr2.score_chipmatch_PL(hs, qcx, chipmatch, qdat)
    elif score_method == 'borda':
        cx2_score, nx2_score = vr2.score_chipmatch_pos(hs, qcx, chipmatch, qdat, 'borda')
    elif score_method == 'topk':
        cx2_score, nx2_score = vr2.score_chipmatch_pos(hs, qcx, chipmatch, qdat, 'topk')
    else:
        raise Exception('[mf] unknown scoring method:' + score_method)
    cx2_nMatch = np.array(map(len, cx2_fm))
    # Autoremove chips with no match support
    cx2_score *= (cx2_nMatch != 0)
    return cx2_score
