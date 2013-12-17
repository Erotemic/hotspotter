
    def get_indexed_uid(hs, with_train=True, with_indx=True):
        indexed_uid = ''
        if with_train:
            indexed_uid += '_trainID(%s)' % hs.train_id
        if with_indx:
            indexed_uid += '_indxID(%s)' % hs.indx_id
        # depends on feat
        indexed_uid += hs.feats.cfg.get_uid()
        return indexed_uid

