import shelve
from other.helpers import filecheck, str2 
from other.AbstractPrintable import AbstractManager
from other.logger import logmsg, logdbg, logio, logerr, logwarn
from pylab import unique
from tpl.pyflann import FLANN
from itertools import chain
from numpy import\
        array, uint32, uint8, empty, ones, float32, log2, savez, load, setdiff1d, int32
from numpy import spacing as eps
# TODO TF-IDF still needs the h or a kmeans to work. 
class VisualModel(AbstractManager):
    '''I know its not really a model, but indexable database 
       isnt a good name
       Indexes the database for fast search'''
    def __init__(vm, hs=None):
        super( VisualModel, vm ).__init__( hs )        
        vm.train_cid    = array([],dtype=uint32) 
        vm.flann = None # This should delete itself
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

    def numWords(vm): 
        return vm.wx2_fdsc.shape[0] #Number of cluster centers
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
        Note: if a cid is assigned to another chip, this will break'''
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
        assert len(tx2_cx) == len(tx2_cid), 'Sanity'
        if len(tx2_cx) == 0:
            logerr('Training set is empty or invalid')
        logdbg('Building model with '+str(vm.num_train())+' sample chips')
        cm.load_features(tx2_cx)
        tx2_nfpts = cm.cx2_nfpts(tx2_cx)
        num_train_keypoints = sum(tx2_nfpts)

        logdbg('Step 2: Build the model Words')
        isTFIDF = False
        if am.model.quantizer == 'none':
            logdbg('No Quantization. Aggregating all fdscriptors for nearest neighbor search.')
            vm.wx2_fdsc = empty((num_train_keypoints,128),dtype=uint8)
            _p = 0
            for cx in tx2_cx:
                nfdsc = cm.cx2_nfpts(cx)
                vm.wx2_fdsc[_p:_p+nfdsc,:] = cm.cx2_fdsc[cx]
                _p += nfdsc
            ax2_wx = array(range(0,num_train_keypoints),dtype=uint32)
            isTFIDF = (am.quantizers['none'].pseudo_num_w > 0)
        if am.model.quantizer == 'hkmeans':
            hkm_cfg = am.quantizers['hkmeans']
            [vm.wx2_fdsc, ax2_wx] = hkmeans_hack([cm.cx2_fdsc[:,tx2_cx]],hkm_cfg)
        if am.model.quantizer == 'akmeans':
            NUM_WORDS = am.quantizers['akmeans'].k
        
        logdbg('Step 3: Point the parts of the model back to their source')
        vm.wx2_axs = empty((vm.numWords()),dtype=object) 
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
        logdbg('Built Model using %d feature vectors. Preparing to index.' % len(vm.ax2_cid))

        logdbg('Step 4: Building FLANN Index: over '+str(len(vm.wx2_fdsc))+' words')
        assert vm.flann is None, 'Flann already exists'
        vm.flann = FLANN()
        flann_param_dict = am.indexers['flann_kdtree'].__dict__
        flann_params = vm.flann.build_index(vm.wx2_fdsc, **flann_param_dict)
        vm.isDirty  = False
        if not vm.save_model():
            logerr('Error Saving Model')

    def save_model(vm): 
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
                nOther   = cm.cx2_num_other_chips(_test_cx)
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
    
