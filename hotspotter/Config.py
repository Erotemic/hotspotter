from __future__ import division, print_function
from hscom.Preferences import Pref
from hscom import helpers

ConfigBase = Pref
#ConfigBase = DynStruct

#=========================
# CONFIG Classes
#=========================
#def udpate_dicts(dict1, dict2):
    #dict1_keys = set(dict1.keys())
    #if key, val in dict2.iteritems():
        #if key in dict1_keys:
            #dict1[key] = val
DEBUG = False

if DEBUG:
    def printDBG(msg):
        print('[DS.DBG] ' + msg)
else:
    def printDBG(msg):
        pass


def dict_subset(dict_, keys):
    'Returns the a subset of the dictionary'
    keys_ = set(keys)
    return {key: val for (key, val)
            in dict_.iteritems() if key in keys_}


def listrm(list_, item):
    'Returns item from list_ if item exists'
    try:
        list_.remove(item)
    except Exception:
        pass


def listrm_list(list_, items):
    'Returns all items in item from list_ if item exists'
    for item in items:
        listrm(list_, item)


#valid_filters = ['recip', 'roidist', 'frexquency', 'ratio', 'bursty', 'lnbnn']
def any_inlist(list_, search_list):
    set_ = set(list_)
    return any([search in set_ for search in search_list])


def signthreshweight_str(on_filters):
    stw_list = []
    for key, val in on_filters.iteritems():
        ((sign, thresh), weight) = val
        stw_str = key
        if thresh is None and weight == 0:
            continue
        if thresh is not None:
            sstr = ['<', '>'][sign == -1]  # actually <=, >=
            stw_str += sstr + str(thresh)
        if weight != 0:
            stw_str += '_' + str(weight)
        stw_list.append(stw_str)
    return ','.join(stw_list)
    #return helpers.remove_chars(str(dict_), [' ','\'','}','{',':'])


class NNConfig(ConfigBase):
    def __init__(nn_cfg, **kwargs):
        super(NNConfig, nn_cfg).__init__()
        # Core
        nn_cfg.K = 4
        nn_cfg.Knorm = 1
        nn_cfg.normalizer_rule = ['last', 'name'][0]
        # Filters
        nn_cfg.checks  = 1024  # 512#128
        nn_cfg.update(**kwargs)

    def get_uid_list(nn_cfg):
        nn_uid  = ['_NN(']
        nn_uid.extend(['K', str(nn_cfg.K)])
        nn_uid.extend(['+', str(nn_cfg.Knorm)])
        nn_uid.extend([',' + nn_cfg.normalizer_rule])
        nn_uid.extend([',cks', str(nn_cfg.checks)])
        nn_uid.extend([')'])
        return nn_uid

    def get_uid(nn_cfg):
        return ''.join(nn_cfg.get_uid_list())


class FilterConfig(ConfigBase):
    # Rename to scoring mechanism
    def __init__(filt_cfg, **kwargs):
        super(FilterConfig, filt_cfg).__init__(name='filt_cfg')
        filt_cfg = filt_cfg
        filt_cfg.filt_on = True
        filt_cfg.Krecip = 0  # 0 := off
        filt_cfg.can_match_sameimg = False
        filt_cfg.can_match_samename = True
        filt_cfg._nnfilter_list = []
        #
        #filt_cfg._nnfilter_list = ['recip', 'roidist', 'lnbnn', 'ratio', 'lnrat']
        filt_cfg._valid_filters = []

        def addfilt(sign, filt, thresh, weight):
            printDBG('[addfilt] %r %r %r %r' % (sign, filt, thresh, weight))
            filt_cfg._nnfilter_list.append(filt)
            filt_cfg._valid_filters.append((sign, filt))
            filt_cfg[filt + '_thresh'] = thresh
            filt_cfg[filt + '_weight'] = weight
        #tuple(Sign, Filt, ValidSignThresh, ScoreMetaWeight)
        # thresh test is: sign * score <= sign * thresh
        addfilt(+1, 'roidist', None, 0)  # Lower  scores are better
        addfilt(-1, 'recip',     0, 0)  # Higher scores are better
        addfilt(+1, 'bursty', None, 0)  # Lower  scores are better
        addfilt(-1, 'ratio',  None, 0)  # Higher scores are better
        addfilt(-1, 'lnbnn',  None, .01)  # Higher scores are better
        addfilt(-1, 'lnrat',  None, 0)  # Higher scores are better
        #addfilt(+1, 'scale' )
        filt_cfg._filt2_tw = {}
        filt_cfg.update(**kwargs)

    def make_feasible(filt_cfg, query_cfg):
        '''
        removes invalid parameter settings over all cfgs (move to QueryConfig)
        '''
        # Ensure the list of on filters is valid given the weight and thresh
        if filt_cfg.ratio_thresh <= 1:
            filt_cfg.ratio_thresh = None
        if filt_cfg.roidist_thresh >= 1:
            filt_cfg.roidist_thresh = None
        if filt_cfg.bursty_thresh   <= 1:
            filt_cfg.bursty_thresh = None
        # FIXME: Non-Independent parameters.
        # Need to explicitly model correlation somehow
        if filt_cfg.Krecip == 0:
            filt_cfg.recip_thresh = None
        elif filt_cfg.recip_thresh is None:
            filt_cfg.recip_thresh = 0
        #print('[cfg]----')
        #print(filt_cfg)
        #print('[cfg]----')

        def _ensure_filter(filt, sign):
            '''ensure filter in the list if valid else remove
            (also ensure the sign/thresh/weight dict)'''
            thresh = filt_cfg[filt + '_thresh']
            weight = filt_cfg[filt + '_weight']
            stw = ((sign, thresh), weight)
            filt_cfg._filt2_tw[filt] = stw
            if thresh is None and weight == 0:
                listrm(filt_cfg._nnfilter_list, filt)
            elif not filt in filt_cfg._nnfilter_list:
                filt_cfg._nnfilter_list += [filt]
        for (sign, filt) in filt_cfg._valid_filters:
            _ensure_filter(filt, sign)
        # Set Knorm to 0 if there is no normalizing filter on.
        if query_cfg is not None:
            nn_cfg = query_cfg.nn_cfg
            norm_depends = ['lnbnn', 'ratio', 'lnrat']
            if nn_cfg.Knorm <= 0 and not any_inlist(filt_cfg._nnfilter_list, norm_depends):
                #listrm_list(filt_cfg._nnfilter_list , norm_depends)
                # FIXME: Knorm is not independent of the other parameters.
                # Find a way to make it independent.
                nn_cfg.Knorm = 0

    def get_uid_list(filt_cfg):
        if not filt_cfg.filt_on:
            return ['_FILT()']
        on_filters = dict_subset(filt_cfg._filt2_tw,
                                 filt_cfg._nnfilter_list)
        filt_uid = ['_FILT(']
        twstr = signthreshweight_str(on_filters)
        if filt_cfg.Krecip != 0 and 'recip' in filt_cfg._nnfilter_list:
            filt_uid += ['Kr' + str(filt_cfg.Krecip)]
            if len(twstr) > 0:
                filt_uid += [',']
        if len(twstr) > 0:
            filt_uid += [twstr]
        if filt_cfg.can_match_sameimg:
            filt_uid += 'same_img'
        if not filt_cfg.can_match_samename:
            filt_uid += 'notsame_name'
        filt_uid += [')']
        return filt_uid

    def get_uid(filt_cfg):
        return ''.join(filt_cfg.get_uid_list())


class SpatialVerifyConfig(ConfigBase):
    def __init__(sv_cfg, **kwargs):
        super(SpatialVerifyConfig, sv_cfg).__init__(name='sv_cfg')
        sv_cfg.scale_thresh_low = .5
        sv_cfg.scale_thresh_high = 2
        sv_cfg.xy_thresh = .01
        sv_cfg.nShortlist = 1000
        sv_cfg.prescore_method = 'csum'
        sv_cfg.use_chip_extent = False
        sv_cfg.just_affine = False
        sv_cfg.min_nInliers = 4
        sv_cfg.sv_on = True
        sv_cfg.update(**kwargs)

    def get_uid_list(sv_cfg):
        if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
            return ['_SV()']
        sv_uid = ['_SV(']
        sv_uid += [str(sv_cfg.nShortlist)]
        sv_uid += [',' + str(sv_cfg.xy_thresh)]
        scale_thresh = (sv_cfg.scale_thresh_low, sv_cfg.scale_thresh_high)
        scale_str = helpers.remove_chars(str(scale_thresh), ' ()')
        sv_uid += [',' + scale_str.replace(',', '_')]
        sv_uid += [',cdl' * sv_cfg.use_chip_extent]  # chip diag len
        sv_uid += [',aff' * sv_cfg.just_affine]  # chip diag len
        sv_uid += [',' + sv_cfg.prescore_method]
        sv_uid += [')']
        return sv_uid

    def get_uid(sv_cfg):
        return ''.join(sv_cfg.get_uid_list())


class AggregateConfig(ConfigBase):
    def __init__(agg_cfg, **kwargs):
        super(AggregateConfig, agg_cfg).__init__(name='agg_cfg')
        agg_cfg.query_type   = 'vsmany'
        # chipsum, namesum, placketluce
        agg_cfg.isWeighted = False  # nsum, pl
        agg_cfg.score_method = 'csum'  # nsum, pl
        alt_methods = {
            'topk': 'topk',
            'borda': 'borda',
            'placketluce': 'pl',
            'chipsum': 'csum',
            'namesum': 'nsum',
        }
        # For Placket-Luce
        agg_cfg.max_alts = 1000
        #-----
        # User update
        agg_cfg.update(**kwargs)
        # ---
        key = agg_cfg.score_method.lower()
        # Use w as a toggle for weighted mode
        if key.find('w') == len(key) - 1:
            agg_cfg.isWeighted = True
            key = key[:-1]
            agg_cfg.score_method = key
        # Sanatize the scoring method
        if key in alt_methods:
            agg_cfg.score_method = alt_methods[key]

    def get_uid_list(agg_cfg):
        agg_uid = []
        agg_uid += ['_AGG(']
        agg_uid += [agg_cfg.query_type]
        agg_uid += [',', agg_cfg.score_method]
        if agg_cfg.isWeighted:
            agg_uid += ['w']
        if agg_cfg.score_method  == 'pl':
            agg_uid += [',%d' % (agg_cfg.max_alts,)]
        agg_uid += [')']
        return agg_uid

    def get_uid(agg_cfg):
        return ''.join(agg_cfg.get_uid_list())


class QueryConfig(ConfigBase):
    def __init__(query_cfg, hs=None, **kwargs):
        super(QueryConfig, query_cfg).__init__(name='query_cfg')
        query_cfg.nn_cfg   = NNConfig(**kwargs)
        query_cfg.filt_cfg = FilterConfig(**kwargs)
        query_cfg.sv_cfg   = SpatialVerifyConfig(**kwargs)
        query_cfg.agg_cfg  = AggregateConfig(**kwargs)
        # Queries depend on features # creating without hs delays crash
        query_cfg._feat_cfg = FeatureConfig(**kwargs) if hs is None else hs.prefs.feat_cfg
        query_cfg.use_cache = False
        if hs is not None:
            query_cfg.update_cfg(**kwargs)

    def update_cfg(query_cfg, **kwargs):
        query_cfg._feat_cfg.update(**kwargs)
        query_cfg._feat_cfg._chip_cfg.update(**kwargs)
        query_cfg.nn_cfg.update(**kwargs)
        query_cfg.filt_cfg.update(**kwargs)
        query_cfg.sv_cfg.update(**kwargs)
        query_cfg.agg_cfg.update(**kwargs)
        query_cfg.update(**kwargs)
        query_cfg.filt_cfg.make_feasible(query_cfg)

    def get_uid_list(query_cfg, *args, **kwargs):
        if query_cfg._feat_cfg is None:
            raise Exception('Feat / chip config is required')
        uid_list = []
        if not 'noNN' in args:
            uid_list += query_cfg.nn_cfg.get_uid_list(**kwargs)
        if not 'noFILT' in args:
            uid_list += query_cfg.filt_cfg.get_uid_list(**kwargs)
        if not 'noSV' in args:
            uid_list += query_cfg.sv_cfg.get_uid_list(**kwargs)
        if not 'noAGG' in args:
            uid_list += query_cfg.agg_cfg.get_uid_list(**kwargs)
        if not 'noCHIP' in args:
            uid_list += query_cfg._feat_cfg.get_uid_list()
        return uid_list

    def get_uid(query_cfg, *args, **kwargs):
        uid_list = query_cfg.get_uid_list(*args, **kwargs)
        uid = ''.join(uid_list)
        return uid


class FeatureConfig(ConfigBase):
    def __init__(feat_cfg, hs=None, **kwargs):
        super(FeatureConfig, feat_cfg).__init__(name='feat_cfg')
        feat_cfg.feat_type = 'hesaff+sift'
        feat_cfg.whiten = False
        feat_cfg.scale_min = 0  # 0  # 30 # TODO: Put in pref types here
        feat_cfg.scale_max = 9001  # 9001 # 80
        feat_cfg.use_adaptive_scale = False  # 9001 # 80
        if hs is not None:
            feat_cfg._chip_cfg = hs.prefs.chip_cfg  # Features depend on chips
        else:
            feat_cfg._chip_cfg = ChipConfig(**kwargs)  # creating without hs delays crash
        feat_cfg.update(**kwargs)

    def get_dict_args(feat_cfg):
        dict_args = {
            'scale_min': feat_cfg.scale_min,
            'scale_max': feat_cfg.scale_max,
            'use_adaptive_scale': feat_cfg.use_adaptive_scale
        }
        return dict_args

    def get_uid_list(feat_cfg):
        if feat_cfg._chip_cfg is None:
            raise Exception('Chip config is required')
        if feat_cfg.scale_min < 0:
            feat_cfg.scale_min = None
        if feat_cfg.scale_max < 0:
            feat_cfg.scale_max = None
        feat_uids = ['_FEAT(']
        feat_uids += [feat_cfg.feat_type]
        feat_uids += [',white'] * feat_cfg.whiten
        feat_uids += [',%r_%r' % (feat_cfg.scale_min, feat_cfg.scale_max)]
        feat_uids += [',adaptive'] * feat_cfg.use_adaptive_scale
        feat_uids += [')']
        feat_uids += feat_cfg._chip_cfg.get_uid_list()
        return feat_uids

    def get_uid(feat_cfg):
        return ''.join(feat_cfg.get_uid_list())


class ChipConfig(ConfigBase):
    def __init__(cc_cfg, **kwargs):
        super(ChipConfig, cc_cfg).__init__(name='chip_cfg')
        cc_cfg.chip_sqrt_area = 750
        cc_cfg.grabcut         = False
        cc_cfg.histeq          = False
        cc_cfg.adapteq         = False
        cc_cfg.region_norm     = False
        cc_cfg.rank_eq         = False
        cc_cfg.local_eq        = False
        cc_cfg.maxcontrast     = False
        cc_cfg.update(**kwargs)

    def get_uid_list(cc_cfg):
        chip_uid = []
        chip_uid += ['histeq']  * cc_cfg.histeq
        chip_uid += ['adapteq'] * cc_cfg.adapteq
        chip_uid += ['grabcut'] * cc_cfg.grabcut
        chip_uid += ['regnorm'] * cc_cfg.region_norm
        chip_uid += ['rankeq']  * cc_cfg.rank_eq
        chip_uid += ['localeq'] * cc_cfg.local_eq
        chip_uid += ['maxcont'] * cc_cfg.maxcontrast
        isOrig = cc_cfg.chip_sqrt_area is None or cc_cfg.chip_sqrt_area  <= 0
        chip_uid += ['szorig'] if isOrig else ['sz%r' % cc_cfg.chip_sqrt_area]
        return ['_CHIP(', (','.join(chip_uid)), ')']

    def get_uid(cc_cfg):
        return ''.join(cc_cfg.get_uid_list())


class DisplayConfig(ConfigBase):
    def __init__(display_cfg, **kwargs):
        super(DisplayConfig, display_cfg).__init__(name='display_cfg')
        display_cfg.N = 6
        display_cfg.name_scoring = False
        display_cfg.showanalysis = False
        display_cfg.annotations  = True
        display_cfg.vert = True  # None
        display_cfg.show_results_in_image = False  # None


# Convinience
def __dict_default_func(dict_):
    # Sets keys only if they dont exist
    def set_key(key, val):
        if not key in dict_:
            dict_[key] = val
    return set_key


def default_display_cfg(**kwargs):
    display_cfg = DisplayConfig(**kwargs)
    return display_cfg


def default_chip_cfg(**kwargs):
    chip_cfg = ChipConfig(**kwargs)
    return chip_cfg


def default_feat_cfg(hs, **kwargs):
    feat_cfg = FeatureConfig(hs, **kwargs)
    return feat_cfg


def default_vsmany_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsmany'
    query_cfg = QueryConfig(hs, **kwargs)
    return query_cfg


def default_vsone_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsone'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', 0)
    kwargs_set('checks', 256)
    kwargs_set('K', 1)
    kwargs_set('Knorm', 1)
    kwargs_set('ratio_weight', 1.0)
    kwargs_set('ratio_thresh', 1.5)
    query_cfg = QueryConfig(hs, **kwargs)
    return query_cfg
