from __future__ import division, print_function
# Python
from os.path import exists, split, join
import os
from zipfile import error as BadZipFile  # Screwy naming convention.
# Scientific
import numpy as np
# HotSpotter
from hscom import helpers
from hscom.Printable import DynStruct
import voting_rules2 as vr2


FM_DTYPE  = np.uint32   # Feature Match datatype
FS_DTYPE  = np.float32  # Feature Score datatype
FK_DTYPE  = np.int16    # Feature Position datatype


#=========================
# Query Result Class
#=========================
def query_result_fpath(hs, qcx, query_uid):
    qres_dir  = hs.dirs.qres_dir
    qcid  = hs.tables.cx2_cid[qcx]
    fname = 'res_%s_qcid=%d.npz' % (query_uid, qcid)
    if len(fname) > 64:
        hash_id = helpers.hashstr(query_uid, 16)
        fname = 'res_%s_qcid=%d.npz' % (hash_id, qcid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(hs, qcx, query_uid):
    fpath = query_result_fpath(hs, qcx, query_uid)
    return exists(fpath)


class QueryResult(DynStruct):
    #__slots__ = ['true_uid', 'qcx', 'query_uid', 'uid', 'title', 'nn_time',
                 #'weight_time', 'filt_time', 'build_time', 'verify_time',
                 #'cx2_fm', 'cx2_fs', 'cx2_fk', 'cx2_score']
    def __init__(res, qcx, uid, qdat=None):
        super(QueryResult, res).__init__()
        res.true_uid  = '' if qdat is None else qdat.get_uid()
        res.qcx       = qcx
        res.query_uid = uid
        res.uid       = uid
        res.title     = uid
        # Assigned features matches
        res.cx2_fm = np.array([], dtype=FM_DTYPE)
        res.cx2_fs = np.array([], dtype=FS_DTYPE)
        res.cx2_fk = np.array([], dtype=FK_DTYPE)
        res.cx2_score = np.array([])
        res.filt2_meta = {}  # messy

    def has_cache(res, hs):
        return query_result_exists(hs, res.qcx)

    def get_fpath(res, hs):
        return query_result_fpath(hs, res.qcx, res.query_uid)

    def save(res, hs):
        fpath = res.get_fpath(hs)
        print('[qr] cache save: %r' % (fpath if hs.args.verbose_cache
                                       else split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            np.savez(file_, **res.__dict__.copy())
        return True

    def load(res, hs):
        'Loads the result from the given database'
        fpath = res.get_fpath(hs)
        print('[res] res.load() fpath=%r' % (split(fpath)[1],))
        qcx_good = res.qcx
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    res.__dict__[_key] = npz[_key]
                npz.close()
            # These are nonarray items even if they are not lists
            # tolist seems to convert them back to their original
            # python representation
            res.qcx = res.qcx.tolist()
            try:
                res.filt2_meta = res.filt2_meta.tolist()
            except AttributeError:
                print('[qr] loading old result format')
                res.filt2_meta = {}
            res.query_uid = str(res.query_uid)
            return True
        except IOError as ex:
            #print('[res] encountered IOError: %r' % ex)
            if not exists(fpath):
                print('[res] cache miss')
                #print(fpath)
                #print('[res] QueryResult(qcx=%d) does not exist' % res.qcx)
                raise
            else:
                msg = ['[res] QueryResult(qcx=%d) is corrupted' % (res.qcx)]
                msg += ['\n%r' % (ex,)]
                print(''.join(msg))
                raise Exception(msg)
        except BadZipFile as ex:
            print('[res] Caught other BadZipFile: %r' % ex)
            msg = ['[res] Attribute Error: QueryResult(qcx=%d) is corrupted' % (res.qcx)]
            msg += ['\n%r' % (ex,)]
            print(''.join(msg))
            if exists(fpath):
                print('[res] Removing corrupted file: %r' % fpath)
                os.remove(fpath)
                raise IOError(msg)
            else:
                raise Exception(msg)
        except Exception as ex:
            print('Caught other Exception: %r' % ex)
            raise
        res.qcx = qcx_good

    def cache_bytes(res, hs):
        fpath = res.get_fpath(hs)
        return helpers.file_bytes(fpath)

    def get_gt_ranks(res, gt_cxs=None, hs=None):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_cxs is None and hs is None:
            raise Exception('[res] error')
        if gt_cxs is None:
            gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        cx2_score = res.get_cx2_score()
        top_cxs  = cx2_score.argsort()[::-1]
        foundpos = [np.where(top_cxs == cx)[0] for cx in gt_cxs]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        gt_ranks = [r[0] for r in ranks_]
        return gt_ranks

    def get_cx2_score(res):
        return res.cx2_score

    def get_cx2_fm(res):
        return res.cx2_fm

    def get_cx2_fs(res):
        return res.cx2_fs

    def get_cx2_fk(res):
        return res.cx2_fk

    def topN_cxs(res, hs, N=None):
        cx2_score = np.array(res.get_cx2_score())
        if hs.prefs.display_cfg.name_scoring:
            cx2_chipscore = np.array(cx2_score)
            cx2_score = vr2.enforce_one_name(hs, cx2_score,
                                             cx2_chipscore=cx2_chipscore)
        top_cxs = cx2_score.argsort()[::-1]
        dcxs_ = set(hs.get_indexed_sample()) - set([res.qcx])
        top_cxs = [cx for cx in iter(top_cxs) if cx in dcxs_]
        #top_cxs = np.intersect1d(top_cxs, hs.get_indexed_sample())
        nIndexed = len(top_cxs)
        if N is None:
            N = hs.prefs.display_cfg.N
        if N == 'all':
            N = nIndexed
        #print('[res] cx2_score = %r' % (cx2_score,))
        #print('[res] returning top_cxs = %r' % (top_cxs,))
        nTop = min(N, nIndexed)
        #print('[res] returning nTop = %r' % (nTop,))
        topN_cxs = top_cxs[0:nTop]
        return topN_cxs

    def show_query(res, hs, **kwargs):
        from hsviz import viz
        print('[res] show_query')
        viz.show_chip(hs, res=res, **kwargs)

    def show_analysis(res, hs, *args, **kwargs):
        from hsviz import viz
        return viz.res_show_analysis(res, hs, *args, **kwargs)

    def show_top(res, hs, *args, **kwargs):
        from hsviz import viz
        return viz.show_top(res, hs, *args, **kwargs)

    def show_gt_matches(res, hs, *args, **kwargs):
        from hsviz import viz
        figtitle = ('q%s -- GroundTruth' % (hs.cidstr(res.qcx)))
        gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        return viz._show_chip_matches(hs, res, gt_cxs=gt_cxs, figtitle=figtitle,
                                      all_kpts=True, *args, **kwargs)

    def show_chipres(res, hs, cx, **kwargs):
        from hsviz import viz
        return viz.res_show_chipres(res, hs, cx, **kwargs)

    def interact_chipres(res, hs, cx, **kwargs):
        from hsviz import interact
        return interact.interact_chipres(hs, res, cx, **kwargs)

    def interact_top_chipres(res, hs, tx, **kwargs):
        from hsviz import interact
        cx = res.topN_cxs(hs, tx + 1)[tx]
        return interact.interact_chipres(hs, res, cx, **kwargs)
