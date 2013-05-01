import shelve
from other.helpers import filecheck, str2 
from other.AbstractPrintable import AbstractManager
from other.logger import logmsg, logdbg, logio, logerr, logwarn
from pylab import unique
from tpl.pyflann import FLANN
from itertools import chain
from numpy import\
        array, uint32, uint8, empty, ones, float32, log2, savez, load, setdiff1d, int32
import numpy as np
from numpy import spacing as eps
# TODO TF-IDF still needs the h or a kmeans to work. 
class VisualModel(AbstractManager):
    #def cx2_visual_word_histogram(vm, cx):
        #fdsc = cm.cx2_fdsc(cx) [fx2_word, dist] = vm.flann_wx2_fdsc.nearest(fdsc) return scipy.sparse_histogram(fx2_word)
        #'''
        #n_clusters : int, optional, default: 8
        #The number of clusters to form as well as the number of centroids to generate.
        #'''

    def build_model2(vm):
        am = vm.hs.am
        logdbg('Build Index was Requested')
        vm.sample_train_set()
        assert len(vm.train_cid) > 0, 'Training set cannot be  empty'
        vm.delete_model()
        train_cx = vm.get_train_cx()
        logdbg('Step 1: Aggregate the model support')
        (ax2_fdsc, ax2_cx, ax2_fx) = vm.aggregate_features(train_cx)
        logdbg('Step 2: Build Whole Chip Representation')
        if am.algo_prefs.model.quantizer == 'naive_bayes':
            wx2_fdsc, wx2_axs = vm.compute_naive_bayes\
                    (ax2_fdsc, ax2_cx, train_cx)
        elif am.algo_prefs.model.quantizer == 'bag_of_words':
            # Get each chip's bag of words,  
            # The words they index into, and the inverse
            wx2_fdsc, wx2_axs, cx2_bow = vm.compute_bag_of_words\
                    (ax2_fdsc, ax2_cx, train_cx)

        logdbg('Step 4: Build Database Representation')
        # Set the model_data = DynStruct
        vm.wx2_fdsc = wx2_fdsc
        vm.wx2_axs  = wx2_axs
        vm.ax2_fx   = ax2_fx
        #vm.ax2_cx    = wx2_axs
        vm.ax2_cid  = vm.hs.cm.cx2_cid[ax2_cx]

        logdbg('Step 5: Building FLANN Index: over '+str(len(vm.wx2_fdsc))+' words')
        assert vm.flann is None, 'Flann already exists'
        vm.flann = FLANN()
        flann_param_dict = am.algo_prefs.model.indexer.to_dict()
        flann_params = vm.flann.build_index(vm.wx2_fdsc, **flann_param_dict)
        vm.isDirty  = False
        if not vm.save_model():
            logerr('Error Saving Model')

    def compute_naive_bayes(vm, ax2_fdsc, ax2_cx, train_cx):
        #wx2_fdsc = ax2_fdsc
        return ax2_fdsc, array([array([ax],dtype=np.uint32) for ax in xrange(len(ax2_fdsc))], dtype=object)
        # Save Words 
        # Save FLANN

    def aggregate_features(vm, train_cx, channel=None):
        # Aggregate Features. Not Saved
        logdbg('Aggregating %d training chips' % len(train_cx))
        cm = vm.hs.cm
        cm.load_features(train_cx)
        # Get how many descriptors each chips has
        tx2_num_fpts = cm.cx2_nfpts(train_cx)
        total_feats = np.sum(tx2_num_fpts)
        ax2_fdsc = empty((total_feats,128), dtype=uint8)
        # Aggregate database descriptors together
        _p = 0
        for tx, cx in enumerate(train_cx):
            num_fpts = tx2_num_fpts[tx]
            ax2_fdsc[_p:_p+num_fpts,:] = cm.cx2_fdsc[cx]
            _p += num_fpts
        # Build Inverted Aggregate Information. (Saved)
        # This needs to be saved as cid
        ax2_cx = np.empty((total_feats,), dtype=np.uint32) 
        ax2_fx = np.empty((total_feats,), dtype=np.uint32)
        ax2_tx = np.empty((total_feats,), dtype=np.uint32)
        _L = 0; _R = 0
        for tx in xrange(len(train_cx)):
            num_fpts    = tx2_num_fpts[tx]
            _R  = _R + num_fpts
            ax_of_tx = np.arange(_L, _R, dtype=np.uint32)
            _L = _L + num_fpts
            ax2_tx[ax_of_tx] = tx
            ax2_cx[ax_of_tx] = train_cx[tx]    # to ChipID
            ax2_fx[ax_of_tx] = np.arange(num_fpts, dtype=np.uint32) # to FeatID
        return (ax2_fdsc, ax2_cx, ax2_fx)
        #return ax2_fdsc
        #vm.ax2_cx = ax2_cx
        #vm.ax2_fx = ax2_cx

    def compute_bag_of_words(vm, ax2_fdsc, ax2_cx, train_cx):
        '''Input: 
            ax2_fdsc - aggregate index to raw descriptor
            ax2_cid  - corresponding raw fdsc's chip id
        '''
        # Parameters
        from back.algo.clustering import approximate_kmeans
        import scipy.sparse
        import collections
        am = vm.hs.am
        total_feats = ax2_fdsc.shape[0]
        num_train = len(train_cx)
        max_iters = am.algo_prefs.model.akmeans.max_iters
        num_words = am.algo_prefs.model.akmeans.num_words
        num_words = min(total_feats,num_words)
        # compute visual_vocab with num_words using akmeans
        # cluster the raw descriptors into visual words
        # NumWords x 128
        akmeans_flann = None
        # Compute Visual Words, and Database Asignments
        wx2_fdsc, wx2_axs = approximate_kmeans\
                (ax2_fdsc, num_words,\
                 max_iters, akmeans_flann)
        # Create Visual Histograms. 
        cx2_tx = {} # For Inverted File
        for tx, cx in enumerate(train_cx):
            cx2_tx[cx] = tx
        sparse_rows = np.empty((total_feats,), dtype=np.uint32)
        sparse_cols = np.empty((total_feats,), dtype=np.uint32)
        rcx = 0
        # Assemble Non-Sparse Histogram Data
        for wx in xrange(num_words):#wx = col
            cxs = [ax2_cx[ax] for ax in wx2_axs[wx]] # txs of cxs = rows
            for cx in cxs:
                tx = cx2_tx[cx]
                sparse_rows[rcx] = tx
                sparse_cols[rcx] = wx
                rcx += 1
        # Build Sparse Vector
        sparse_data = np.ones(total_feats, dtype=uint8)
        cx2_bow = scipy.sparse.coo_matrix\
                ((sparse_data, (sparse_rows, sparse_cols)), dtype=uint32)
        return wx2_fdsc, wx2_axs, cx2_bow

    def compute_tfidf():
        pass
        #from collections import Counter
        #term_freq_pairs = Counter(cxs).items()
        #unique_cx, ucx2_term_frequency = apply(zip, term_freq_pairs)
        #wx2_termfreq = sum(ucx2_term_frequency)
        #cx2_tfidf[unique_cx, wx] = ucx2_term_frequency
        #cx2_tfidf  = scipy.sparse.coo_matrix\
                #((num_train, num_words), dtype=uint32)

        #logdbg('Computing TF-IDF metadata')
        #bcarg = {'max_tx':len(tx2_cx), 'dtype':np.float32}
        #tx2_termfq_denom = float32(cm.cx2_nfpts(tx2_cx))
        #vm.wx2_maxtf = \
                #[ max( bincount(ax2_tx[axs], **bcarg)/termfq_denom ) for axs in vm.wx2_axs]
        #TODO: 
        # timeit [ np.max( bincount(ax2_tx[axs], **bcarg)/termfq_denom ) for axs in vm.wx2_axs]
        # timeit x = [ bincount(ax2_tx[axs], **bcarg)/termfq_denom for axs in vm.wx2_axs]
        #        [ np.max(y) for y in x ]
        # timeit max
        #num_train = vm.num_train()
        #vm.wx2_idf = log2([ num_train/len(unique(ax2_tx[ax_of_wx])) for ax_of_wx in vm.wx2_axs ] )+eps(1)
        #vm.wx2_fdsc = sklearn.cluster.KMeans(
        # Non-quantized vocabulary

    def compute_inverted_file(vm, wx2_fdsc, train_cx):
        cm = vm.hs.cm
        wx2_axs = alloc_list(wx2_fdsc.shape[0])
        for ax in xrange(0, len(ax2_fx)):
            if vm.wx2_axs[ax] is None:
                vm.wx2_axs[ax] = []
            wx = ax2_wx[ax]
            vm.wx2_axs[wx].append(ax)
        ax2_cid = -ones(len(train_cx),dtype=int32) 
        ax2_fx  = -ones(len(train_cx),dtype=int32)
        ax2_tx  = -ones(len(train_cx),dtype=int32)
        _Lfx = 0; _Rfx = 0
        tx2_nfpts = cm.cx2_nfpts(train_cx)
        for tx in xrange(train_cx):
            nfpts    = tx2_nfpts[tx]
            next_fx  = _Rfx + nfpts
            ax_range = range(_Lfx,_Rfx)
            ax2_tx[ax_range] = tx
            ax2_cid[ax_range] = cm.cx2_cid[train_cx]  # Point to Inst
            _Lfx = _Rfx + nfpts
        if isTFIDF: # Compute info for TF-IDF
            logdbg('Computing TF-IDF metadata')
            max_tx = len(tx2_cx)
            tx2_wtf_denom = float32(cm.cx2_nfpts(tx2_cx))
            vm.wx2_maxtf = map(lambda ax_of_wx:\
                max( float32(bincount(ax2_tx[ax_of_wx], minlength=max_tx)) / tx2_wtf_denom ), vm.wx2_axs)
            vm.wx2_idf = log2(map(lambda ax_of_wx:\
                vm.num_train()/len(unique(ax2_tx[ax_of_wx])),\
                vm.wx2_axs)+eps(1))
        logdbg('Built Model using %d feature vectors. Preparing to index.' % len(vm.ax2_cid))


        vm.wx2_axs = empty((vm.wx2_fdsc.shape[0]),dtype=object) 
        for ax in xrange(0,num_train_keypoints):
            if vm.wx2_axs[ax] is None:
                vm.wx2_axs[ax] = []
            wx = ax2_wx[ax]
            vm.wx2_axs[wx].append(ax)
        vm.ax2_cid = -ones(num_train_keypoints,dtype=int32) 
        vm.ax2_fx  = -ones(num_train_keypoints,dtype=int32)
        ax2_tx     = -ones(num_train_keypoints,dtype=int32)
        curr_fx = 0; next_fx = 0
        for tx in xrange(vm.num_train()):
            nfpts    = tx2_nfpts[tx]
            next_fx  = next_fx + nfpts
            ax_range = range(curr_fx,next_fx)
            ax2_tx[ax_range] = tx
            vm.ax2_cid[ax_range] = tx2_cid[tx]    # Point to Inst
            vm.ax2_fx[ax_range]  = range(nfpts)   # Point to Kpts
            curr_fx = curr_fx + nfpts
        if isTFIDF: # Compute info for TF-IDF
            logdbg('Computing TF-IDF metadata')
            max_tx = len(tx2_cx)
            tx2_wtf_denom = float32(cm.cx2_nfpts(tx2_cx))
            vm.wx2_maxtf = map(lambda ax_of_wx:\
                max( float32(bincount(ax2_tx[ax_of_wx], minlength=max_tx)) / tx2_wtf_denom ), vm.wx2_axs)
            vm.wx2_idf = log2(map(lambda ax_of_wx:\
                vm.num_train()/len(unique(ax2_tx[ax_of_wx])),\
                vm.wx2_axs)+eps(1))


    def lazyprop(fn):
        # From http://stackoverflow.com/questions/3012421/python-lazy-property-decorator
        attr_name = '_lazy_' + fn.__name__
        @property
        def _lazyprop(self, *args, **kwargs):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, self.fn(*args, **kwargs))
            return getattr(self, attr_name)
        return _lazyprop

    @lazyprop
    def flann_index(data_vecs):
        flann       = FLANN()
        flann_args  = vm.hs.am.algo_prefs.model.indexer.to_dict()
        flann_args_ = flann.build_index(data_vecs, **flann_args)
        return flann

    #, filename=None, cache=False
    def flann_nearest(vm, data_vecs, query_vecs, K): 
        ''' Let N = len(query_vecs) 
            Returns: 
                index_list - (N x K) Index of the Nth query_vec's Kth nearest neighbor
                dist_list - (N x K) Coresponding Distance'''
        N = query_vecs.shape[0]
        flann_index = vm.flann_index(data_vecs)
        (index_list, dist_list) = flann_index.nn_index(query_vecs, K, checks=128)
        index_list.shape = (N, K)
        dist_list.shape = (N, K)
        return (index_list, dist_list)

    def __init__(vm, hs=None):
        super( VisualModel, vm ).__init__( hs )        
        vm.train_cid    = array([],dtype=uint32) 
        vm.flann    = None # This should delete itself
        vm.isDirty  = True 
        # --- The Sample Data ---
        vm.sample_filter = {'exclude_cids'         : [],
                            'one_out_each_name'    : False,
                            'less_than_offset_ok'  : False,
                            'offset'               : 1,
                            'randomize'            : 0,
                            'max_numc_per_name'    :-1,
                            'min_numc_per_name'    :-1,
                            }
        # --- Bookkeeping --
        vm.exclude_load_mems = ['hs', 'flann', 'isDirty', 'exclude_load_mems', 'sample_filter']
        
    def reset(vm):
        logmsg('Reseting the Visual Model')
        # ---
        vm.isDirty  = True 
        # --- Inverted Index ---
        # The support for the model (aka visual word custer centers)
        # In the case of Naive Bayes, this is the raw features
        # In the case of Quantization, these are the cluster centers
        vm.wx2_fdsc   = array([], dtype=uint8) 
        vm.wx2_axs    = []  # List of axs belonging to this word 
        # --- TFIDF-Model ---
        vm.wx2_idf    = array([]) # Word -> Inverse Document Frequency
        vm.wx2_maxtf  = array([]) # Word -> Maximum Database Term Frequency
        # --- Model Source Metadata --
        vm.ax2_cid    = array([], dtype=uint32) # indexed chips
        vm.ax2_fx     = array([], dtype=uint32) # indexed features

    def num_train(vm):
        return len(vm.train_cid)
    def  get_train_cx(vm):
        return vm.hs.cm.cid2_cx[vm.get_train_cid()]
    def  get_test_cx(vm):
        return setdiff1d(vm.hs.cm.all_cxs(), vm.get_train_cx())
    def  get_test_cid(vm):
        return vm.hs.cm.cx2_cid[vm.get_test_cx()]
    def  get_train_cid(vm):
        return vm.train_cid
    def  flip_sample(vm):
        vm.train_cid = vm.get_test_cid()

    def  get_samp_id(vm):
        ''''Returns an id unique to the sampled train_cid
        Note: if a cid is assigned to another.hip, this will break'''
        iom = vm.hs.iom
        samp_shelf = shelve.open(iom.get_temp_fpath('sample_shelf.db'))
        samp_key = '%r' % vm.train_cid
        if not samp_key in samp_shelf.keys():
            samp_shelf[samp_key] = len(samp_shelf.keys())+1
        samp_id = samp_shelf[samp_key]
        samp_shelf.close()
        return samp_id
    def  get_samp_suffix(vm):
        return '.samp'+str(vm.get_samp_id())
    def ax2_cx(vm, axs):
        return vm.hs.cm.cid2_cx[vm.ax2_cid[axs]]
    def delete_model(vm):
        logdbg('Deleting Sample Index')
        if vm.flann != None:
            try:
                vm.flann.delete_index()
                vm.flann = None
            except WindowsError: 
                logwarn('WARNING: FLANN is not deleting correctly')
        vm.reset()

# OLD OLD OLD
    def nearest_neighbors(vm, qfdsc, K): 
        ''' qfx2_wxs - (num_feats x K) Query Descriptor Index to the K Nearest Word Indexes 
            qfx2_dists - (num_feats x K) Query Descriptor Index to the Distance to the  K Nearest Word Vectors '''
        assert vm.flann is not None  , 'Cant query empty index'
        assert len(qfdsc) != 0       , 'Cant have empty query'
        logdbg('Searching for Nearest Neighbors: #vectors=%d, K=%d' % (len(qfdsc), K))
        (qfx2_Kwxs, qfx2_Kdists) = vm.flann.nn_index(qfdsc, K, checks=128)
        qfx2_Kwxs.shape   =  (qfdsc.shape[0], K)
        qfx2_Kdists.shape =  (qfdsc.shape[0], K)
        return (qfx2_Kwxs, qfx2_Kdists)


    #Probably will have to make this over cids eventually\ Maybe
    def build_model(vm, force_recomp=False):
        if not force_recomp and not vm.isDirty:
            logmsg('The model is clean and is not forced to recompute')
            return True
        logmsg('Building the model. If you have over 1000 chips, this will take awhile and there may be no indication of progress.')
        am = vm.hs.am
        cm = vm.hs.cm
        logdbg('Build Index was Requested')
        vm.delete_model()
        vm.sample_train_set()
        logdbg('Step 1: Aggregate the model support (Load feature vectors) ---')
        tx2_cx   = vm.get_train_cx()
        tx2_cid  = vm.get_train_cid()
        assert len(tx2_cx) > 0, 'Training set cannot be  empty'
        logdbg('Building model with %d sample chips' % (vm.num_train()))
        cm.load_features(tx2_cx)
        tx2_nfpts = cm.cx2_nfpts(tx2_cx)
        num_train_keypoints = sum(tx2_nfpts)

        logdbg('Step 2: Build the model Words')
        isTFIDF = False
        if am.algo_prefs.model.quantizer == 'naive_bayes':
            logdbg('No Quantization. Aggregating all fdscriptors for nearest neighbor search.')
            vm.wx2_fdsc = empty((num_train_keypoints,128),dtype=uint8)
            _p = 0
            for cx in tx2_cx:
                nfdsc = cm.cx2_nfpts(cx)
                vm.wx2_fdsc[_p:_p+nfdsc,:] = cm.cx2_fdsc[cx]
                _p += nfdsc
            ax2_wx = array(range(0,num_train_keypoints),dtype=uint32)
        if am.algo_prefs.model.quantizer == 'akmeans':
            raise NotImplementedError(':)')
        
        logdbg('Step 3: Point the parts of the model back to their source')
        vm.wx2_axs = empty(vm.wx2_fdsc.shape[0], dtype=object) 
        for ax in xrange(0,num_train_keypoints):
            if vm.wx2_axs[ax] is None:
                vm.wx2_axs[ax] = []
            wx = ax2_wx[ax]
            vm.wx2_axs[wx].append(ax)
        vm.ax2_cid = -ones(num_train_keypoints,dtype=np.int32) 
        vm.ax2_fx  = -ones(num_train_keypoints,dtype=np.int32)
        ax2_tx     = -ones(num_train_keypoints,dtype=np.int32)
        curr_fx = 0; next_fx = 0
        for tx in xrange(vm.num_train()):
            nfpts    = tx2_nfpts[tx]
            next_fx  = next_fx + nfpts
            ax_range = range(curr_fx,next_fx)
            ax2_tx[ax_range] = tx
            vm.ax2_cid[ax_range] = tx2_cid[tx]    # Point to Inst
            vm.ax2_fx[ax_range]  = range(nfpts)   # Point to Kpts
            curr_fx = curr_fx + nfpts
        if isTFIDF: # Compute info for TF-IDF
            logdbg('Computing TF-IDF metadata')
            max_tx = len(tx2_cx)
            tx2_wtf_denom = float32(cm.cx2_nfpts(tx2_cx))
            vm.wx2_maxtf = map(lambda ax_of_wx:\
                max( float32(bincount(ax2_tx[ax_of_wx], minlength=max_tx)) / tx2_wtf_denom ), vm.wx2_axs)
            vm.wx2_idf = log2(map(lambda ax_of_wx:\
                vm.num_train()/len(unique(ax2_tx[ax_of_wx])),\
                vm.wx2_axs)+eps(1))
        logdbg('Built Model using %d feature vectors. Preparing to index.' % len(vm.ax2_cid))

        logdbg('Step 4: Building FLANN Index: over '+str(len(vm.wx2_fdsc))+' words')
        assert vm.flann is None, 'Flann already exists'
        vm.flann = FLANN()
        flann_param_dict = am.algo_prefs.model.indexer.to_dict()
        flann_params = vm.flann.build_index(vm.wx2_fdsc, **flann_param_dict)
        vm.isDirty  = False
        #if not vm.save_model():
            #logerr('Error Saving Model')

    def save_model(vm): 
        raise NotImplementedError()
        iom = vm.hs.iom
        model_fpath = iom.get_model_fpath()
        flann_index_fpath  = iom.get_flann_index_fpath()
        #Crazy list compreshension. 
        #Returns (key,val) tuples for saveable vars
        if vm.isDirty:
            raise Exception('Cannot Save a Dirty Index')
        to_save_dict = {key : vm.__dict__[key] \
                             for key in vm.__dict__.keys() \
                             if key.find('__') == -1 and key not in vm.exclude_load_mems }
        logio('Saving Model')
        savez(model_fpath, **to_save_dict)
        logio('Saved Model to '+model_fpath)
        assert not vm.flann is None, 'Trying to save null flann index'
        vm.flann.save_index(flann_index_fpath)
        logio('Saved model index to '+flann_index_fpath)
        return True

    def _save_class(vm, cls):
        to_save_dict = {key : vm.__dict__[key] \
                             for key in vm.__dict__.keys() \
                             if key.find('__') == -1 and key not in vm.exclude_load_mems }
    def _load_class(vm):
        pass

    def load_model(vm):
        raise NotImplementedError()
        vm.delete_model()
        logio('Checking for previous model computations')
        iom = vm.hs.iom
        model_fpath = iom.get_model_fpath()
        flann_index_fpath = iom.get_flann_index_fpath()

        if not (filecheck(model_fpath) and filecheck(flann_index_fpath)):
            return False
        logio('Reading Precomputed Model: %s\nKey/Vals Are:' % model_fpath)
        npz = load(model_fpath)
        tmpdict = {}
        for _key in npz.files:
            _val = npz[_key]
            tmpdict[_key] = _val
            logdbg('  * <'+str2(type(_val))+'> '+str(_key)+' = '+str2(_val))
        npz.close()

        for _key in tmpdict.keys():
            vm.__dict__[_key] = tmpdict[_key]
        logio('Reading Precomputed FLANN Index: %s' % flann_index_fpath)
        assert vm.flann is None, '! Flann already exists'
        vm.flann = FLANN()
        vm.flann.load_index(flann_index_fpath, vm.wx2_fdsc)
        vm.isDirty = True
        logmsg('The model is built')
        return True
    
    def sample_train_set(vm, samp_filter_arg=None):
        cm = vm.hs.cm; nm = vm.hs.nm
        if samp_filter_arg is None: 
            filt = vm.sample_filter
        else: 
            filt = samp_filter_arg
        logdbg('Collecting sample set: \n  SAMP_FILTER:\n    - '+str(filt).replace(', ',' \n    - ')[1:-1])
        old_train = vm.train_cid
        train_cx = cm.all_cxs()
        #Filter things out from vm.train_cx
        if filt['min_numc_per_name'] > -1:
            _min     = filt['min_numc_per_name']
            vnxs     = nm.get_valid_nxs(min_chips=_min)
            cxsPool  = [nm.nx2_cx_list[_cx] for _cx in vnxs]
            train_cx = list(chain.from_iterable(cxsPool))
        if filt['one_out_each_name'] is True:
            vnxs     = nm.get_valid_nxs()
            offset   = filt['offset']
            cxsPool  = [nm.nx2_cx_list[_cx] for _cx in vnxs]
            pickFun  = lambda cxs: offset % len(cxs)
            _test_cx = array(map(lambda cxs: cxs[pickFun(cxs)], cxsPool))
            if samp_filter_arg['less_than_offset_ok'] is False:
                nOther   = cm.cx2_num_other.hips(_test_cx)
                _okBit   = nOther > offset
                _test_cx = _test_cx[_okBit]
            train_cx = setdiff1d(vm.train_cx, _test_cx)
        if len(filt['exclude_cids']) > 0:
            exclu_cx = cm.cx2_cid[filt['exclude_cids']]
            train_cx = setdiff1d(train_cx, exclu_cx)
        vm.train_cid = cm.cx2_cid[train_cx]
        logdbg('Train: '+str(vm.get_train_cid()))
        logdbg('Test: '+str(vm.get_test_cid()))
        if not vm.isDirty:
            if not (len(old_train) == len(vm.train_cid)\
                    and all(old_train == vm.train_cid)):
                vm.isDirty = True
                logdbg('The sample has changed.')
            else:
                logdbg('The sample has not changed.')
        logdbg('The index is '+['Clean','Dirty'][vm.isDirty])

        """
    def lazy_computation(fn, fpath):
        def lazy_wrapper(*args, **kwargs):
            global lazy
            if fpath in lazy.loaded.keys():
                lazyvar = lazy.loaded[fpath]
            elif fpath in lazy.canload(fpath):
                lazyvar = lazy.load(fpath)
            else:
                lazyvar = fn(*args, **kwargs)
                lazy.loaded[fpath] = lazyvar
            return lazyvar
        return lazy_wrapper
    def lazy_savefunc(fn, fpath):
        global lazy
        lazy.savefuncs[fpath] = fn
    def lazy_loadfunc(fn, fpath):
        global lazy
        lazy.loadfuncs[fpath] = fn
    @lazy_savefunc('C:/tmp.flann'):
    def _flann_save():
        vm.flann.save_index(flann_index_fpath)
    @lazy_loadfunc('C:/tmp.flann'):
    def _flann_load(data_vecs):
        vm.flann = FLANN()
        vm.flann.load_index(flann_index_fpath, vm.wx2_fdsc)
    @lazy_computation('C:/tmp.flann')
    """
