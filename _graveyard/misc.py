def reload_module():
    import imp, sys
    print('[misc] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def dev_correct_num_words(db_target, db_dir):
    # function to get correct num words per descriptor based on oxford
    from __init__ import *
    params.rrr()
    def ndesc_of_db(db_dir):
        hs = ld2.HotSpotter()
        hs.load_tables(db_dir)
        hs.load_chips()
        hs.load_features(load_desc=False)
        cx2_nkpts = map(len, hs.feats.cx2_kpts)
        ndesc = sum(cx2_nkpts)
        print(db_dir+' ndesc=%r' % ndesc)
        return ndesc 
    gz_ndesc = ndesc_of_db(params.GZ)
    pz_ndesc = ndesc_of_db(params.PZ)
    mothers_ndesc = ndesc_of_db(params.MOTHERS)
    params.oxford_defaults()
    ox_ndesc = ndesc_of_db(params.OXFORD)
    ndesc_per_word = ox_ndesc / 1000000 
    gz_nwords      = gz_ndesc / ndesc_per_word
    pz_ndesc       = pz_ndesc / ndesc_per_word
    mothers_nwords = mothers_ndesc / ndesc_per_word
    print('GZ num words: %r ' % round(gz_nwords) )
    print('PZ num words: %r ' % round(pz_ndesc) )
    print('MO num words: %r ' % round(mothers_nwords) )
