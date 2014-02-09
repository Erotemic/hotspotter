#=========================
# CONFIG Classes
#=========================
#def udpate_dicts(dict1, dict2):
    #dict1_keys = set(dict1.keys())
    #if key, val in dict2.iteritems():
        #if key in dict1_keys:
            #dict1[key] = val


#def dict_subset(dict_, keys):
    #'Returns the a subset of the dictionary'
    #keys_ = set(keys)
    #return {key: val for (key, val)
            #in dict_.iteritems() if key in keys_}


#def listrm(list_, item):
    #'Returns item from list_ if item exists'
    #try:
        #list_.remove(item)
    #except Exception:
        #pass


#def listrm_list(list_, items):
    #'Returns all items in item from list_ if item exists'
    #for item in items:
        #listrm(list_, item)


#valid_filters = ['recip', 'roidist', 'frexquency', 'ratio', 'bursty', 'lnbnn']
#def any_inlist(list_, search_list):
    #set_ = set(list_)
    #return any([search in set_ for search in search_list])


#def signthreshweight_str(on_filters):
    #stw_list = []
    #for key, val in on_filters.iteritems():
        #((sign, thresh), weight) = val
        #stw_str = key
        #if thresh is None and weight == 0:
            #continue
        #if thresh is not None:
            #sstr = ['<', '>'][sign == -1]  # actually <=, >=
            #stw_str += sstr + str(thresh)
        #if weight != 0:
            #stw_str += '_' + str(weight)
        #stw_list.append(stw_str)
    #return ','.join(stw_list)
    ##return helpers.remove_chars(str(dict_), [' ','\'','}','{',':'])




        # FIXME: Non-Independent parameters.
        # Need to explicitly model correlation somehow
        #if filt_cfg.Krecip == 0:
            #filt_cfg.recip_thresh = None
        #elif filt_cfg.recip_thresh is None:
            #filt_cfg.recip_thresh = 0
        #print('[cfg]----')
        #print(filt_cfg)
        #print('[cfg]----')
        #def _ensure_filter(filt, sign):
            #'''ensure filter in the list if valid else remove
            #(also ensure the sign/thresh/weight dict)'''
            #thresh = filt_cfg[filt + '_thresh']
            #weight = filt_cfg[filt + '_weight']
            #stw = ((sign, thresh), weight)
            #filt_cfg._filt2_tw[filt] = stw
            #if thresh is None and weight == 0:
                #listrm(filt_cfg._nnfilter_list, filt)
            #elif not filt in filt_cfg._nnfilter_list:
                #filt_cfg._nnfilter_list += [filt]
        #for (sign, filt) in filt_cfg._valid_filters:
            #_ensure_filter(filt, sign)
        # Set Knorm to 0 if there is no normalizing filter on.
        #if query_cfg is not None:
            #nn_cfg = query_cfg.nn_cfg
            #norm_depends = ['lnbnn', 'ratio', 'lnrat']
            #if nn_cfg.Knorm <= 0 and not any_inlist(filt_cfg._nnfilter_list, norm_depends):
                #listrm_list(filt_cfg._nnfilter_list , norm_depends)
                # FIXME: Knorm is not independent of the other parameters.
                # Find a way to make it independent.
                #nn_cfg.Knorm = 0

