import shelve
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct, Pref 
from hotspotter.other.logger import logmsg, logdbg, logio, logerr, logwarn
from hotspotter.tpl.pyflann import FLANN
from itertools import chain
import numpy as np
import pylab
import os
from numpy import spacing as eps
# TODO TF-IDF still needs the h or a kmeans to work. 
class VisualModel(AbstractManager):

    def init_preferences(vm, default_bit=False):
        if vm.model_prefs == None:
            pref_fpath = vm.hs.iom.get_prefs_fpath('visual_model_prefs')
            vm.model_prefs = Pref(fpath=pref_fpath)
            vm.model_prefs.save_load_model = Pref(True)


    def __init__(vm, hs=None):
        super( VisualModel, vm ).__init__( hs )        
        vm.model_prefs = None
        # ---
        vm.train_cid = np.array([],dtype=np.uint32) 
        vm.flann = None # This should delete itself
        vm.isDirty  = True 
        # --- Inverted Index ---
        # The support for the model (aka visual word custer centers)
        # In the case of Naive Bayes, this is the raw features
        # In the case of Quantization, these are the cluster centers
        vm.wx2_fdsc   = np.array([], dtype=np.uint8) 
        vm.wx2_axs    = []  # List of axs belonging to this word 
        # --- TFIDF-Model ---
        vm.wx2_idf    = np.array([], dtype=np.float32) # Word -> Inverse Document Frequency
        vm.wx2_maxtf  = np.array([], dtype=np.float32) # Word -> Maximum Database Term Frequency
        # --- Model Source Metadata --
        vm.ax2_cid    = np.array([], dtype=np.uint32) # indexed chips
        vm.ax2_fx     = np.array([], dtype=np.uint32) # indexed features
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
        vm.savable_model_fields = [ 'wx2_fdsc', 'wx2_axs', 'wx2_idf',
                                    'wx2_maxtf', 'ax2_cid', 'ax2_fx',
                                    'train_cid']
        vm.init_preferences()

    def reset(vm):
        logmsg('Reseting the Visual Model')
        vm.isDirty  = True 
        vm.wx2_fdsc   = np.array([], dtype=np.uint8) 
        vm.wx2_axs    = []
        vm.wx2_idf    = np.array([], dtype=np.float32)
        vm.wx2_maxtf  = np.array([], dtype=np.float32)
        vm.ax2_cid    = np.array([], dtype=np.uint32)
        vm.ax2_fx     = np.array([], dtype=np.uint32)

    def ax2_cx(vm, axs):
        'aggregate index to chip index'
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

    # SHOULD BECOME DEPREICATED
    def nearest_neighbors(vm, qfdsc, K): 
        ''' qfx2_wxs - (num_feats x K) Query Descriptor Index to the K Nearest Word Indexes 
            qfx2_dists - (num_feats x K) Query Descriptor Index to the Distance to the  K Nearest Word Vectors '''
        #assert vm.flann is not None  , 'Cant query np.empty index'
        #assert len(qfdsc) != 0       , 'Cant have np.empty query'
        #logdbg('Searching for Nearest Neighbors: #vectors=%d, K=%d' % (len(qfdsc), K))
        (qfx2_Kwxs, qfx2_Kdists) = vm.flann.nn_index(qfdsc, K, checks=128)
        qfx2_Kwxs.shape   =  (qfdsc.shape[0], K)
        qfx2_Kdists.shape =  (qfdsc.shape[0], K)
        return (qfx2_Kwxs, qfx2_Kdists)

    #Probably will have to make this over cids eventually\ Maybe
    def build_model(vm, force_recomp=False):
        ''' Builds the model, if needed. Tries to reload if it can '''
        logmsg('\n\nRequested: Build Model')
        if not force_recomp and not vm.isDirty:
            logmsg('The model is clean and is not forced to recompute')
            return True
        cm = vm.hs.cm
        # Delete old index and resample chips to index
        vm.delete_model()
        vm.sample_train_set()
        # Try to load the correct model
        if not force_recomp and vm.load_model():
            logmsg('Loaded saved model from disk')
            return
        logmsg('Building the model. This may take some time.')
        # Could not load old model. Do full rebuild
        # -----
        # STEP 1 - Loading
        logdbg('Step 1: Aggregate the model support (Load feature vectors) ---')
        tx2_cx   = vm.get_train_cx()
        tx2_cid  = vm.get_train_cid()
        assert len(tx2_cx) > 0, 'Training set cannot be  np.empty'
        logdbg('Building model with %d sample chips' % (vm.num_train()))
        cm.load_features(tx2_cx)
        tx2_nfpts = cm.cx2_nfpts(tx2_cx)
        num_train_keypoints = sum(tx2_nfpts)
        # -----
        # STEP 2 - Aggregating 
        logdbg('Step 2: Build the model Words')
        isTFIDF = False
        if vm.hs.am.algo_prefs.model.quantizer == 'naive_bayes':
            logdbg('No Quantization. Aggregating all fdscriptors for nearest neighbor search.')
            vm.wx2_fdsc = np.empty((num_train_keypoints,128),dtype=np.uint8)
            _p = 0
            for cx in tx2_cx:
                nfdsc = cm.cx2_nfpts(cx)
                vm.wx2_fdsc[_p:_p+nfdsc,:] = cm.cx2_fdsc[cx]
                _p += nfdsc
            ax2_wx = np.array(range(0,num_train_keypoints),dtype=np.uint32)
        if vm.hs.am.algo_prefs.model.quantizer == 'akmeans':
            raise NotImplementedError(':)')
        # -----
        # STEP 3 - Inverted Indexing
        logdbg('Step 3: Point the parts of the model back to their source')
        vm.wx2_axs = np.empty(vm.wx2_fdsc.shape[0], dtype=object) 
        for ax in xrange(0,num_train_keypoints):
            if vm.wx2_axs[ax] is None:
                vm.wx2_axs[ax] = []
            wx = ax2_wx[ax]
            vm.wx2_axs[wx].append(ax)
        vm.ax2_cid = -np.ones(num_train_keypoints,dtype=np.int32) 
        vm.ax2_fx  = -np.ones(num_train_keypoints,dtype=np.int32)
        ax2_tx     = -np.ones(num_train_keypoints,dtype=np.int32)
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
            tx2_wtf_denom = np.float32(cm.cx2_nfpts(tx2_cx))
            vm.wx2_maxtf = map(lambda ax_of_wx:\
                max( np.float32(bincount(ax2_tx[ax_of_wx], minlength=max_tx)) / tx2_wtf_denom ), vm.wx2_axs)
            vm.wx2_idf = np.log2(map(lambda ax_of_wx:\
                vm.num_train()/len(pylab.unique(ax2_tx[ax_of_wx])),\
                vm.wx2_axs)+eps(1))
        logdbg('Built Model using %d feature vectors. Preparing to index.' % len(vm.ax2_cid))
        # -----
        # STEP 4 - Indexing
        logdbg('Step 4: Building FLANN Index: over '+str(len(vm.wx2_fdsc))+' words')
        assert vm.flann is None, 'Flann already exists'
        vm.flann = FLANN()
        flann_param_dict = vm.hs.am.algo_prefs.model.indexer.to_dict()
        flann_params = vm.flann.build_index(vm.wx2_fdsc, **flann_param_dict)
        vm.isDirty  = False
        vm.save_model()
        logmsg('The model was built.')
        

    def save_model(vm):
        # See if the model is savable
        if not vm.model_prefs.save_load_model:
            logdbg('Can NOT save the visual model due to preferences')
            return False
        if vm.isDirty:
            raise Exception('Can NOT save the visual model due to dirty index')
        if vm.flann is None:
            raise Exception('Can NOT save the visual model without a flann index')
        logdbg('Building dictionary to save')
        # TODO: This dictionary should just exist and not be 
        # directly tied to this class.
        # Build a dictionary of savable model terms
        to_save_dict = {key : vm.__dict__[key] \
                        for key in vm.savable_model_fields }
        # Get the save paths
        model_fpath = vm.hs.iom.get_model_fpath()
        flann_index_fpath = vm.hs.iom.get_flann_index_fpath()
        # Save the Model
        logio('Saving model to: '+model_fpath)
        np.savez(model_fpath, **to_save_dict)
        # Save the Index
        logio('Saving index to: '+flann_index_fpath)
        vm.flann.save_index(flann_index_fpath)
        logio('Model save was sucessfull')
        return True

    def load_model(vm):
        # See if the model is loadable
        if not vm.model_prefs.save_load_model:
            logdbg('Can NOT load the visual model')
            return False
        if not vm.flann is None: 
            raise Exception('Cannot load a model when FLANN already exists')
        logdbg('Trying to load visual model')
        # Check to see if new model on disk
        model_fpath = vm.hs.iom.get_model_fpath()
        if not os.path.exists(model_fpath):
            logdbg(' * A saved model data file was missing: '+
                   model_fpath); return False
        flann_index_fpath = vm.hs.iom.get_flann_index_fpath()
        if not os.path.exists(flann_index_fpath):
            logdbg(' * A saved flann index file was missing: '+
                   flann_index_fpath); return False
        # Model and Flann Exist on disk
        # Load the model data first
        # Read model into dictionary
        logmsg('Loading visual model data: ' + model_fpath)
        npz = np.load(model_fpath)
        for _key in npz.files:
            vm.__dict__[_key] = npz[_key]
        npz.close()
        # Read FLANN index
        logmsg('Loading FLANN index: '+ flann_index_fpath)
        vm.flann = FLANN()
        vm.flann.load_index(flann_index_fpath, vm.wx2_fdsc)
        vm.isDirty = False
        logmsg('The model was sucessfully loaded')
        return True

    # SHOULD BECOME DEPRICATED
    def sample_train_set(vm, samp_filter_arg=None):
        ''' This is some pretty legacy matlab stuff. It builds a sample set for
        the model based on some specifications'''
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
            _test_cx = np.array(map(lambda cxs: cxs[pickFun(cxs)], cxsPool))
            if samp_filter_arg['less_than_offset_ok'] is False:
                nOther   = cm.cx2_num_other.hips(_test_cx)
                _okBit   = nOther > offset
                _test_cx = _test_cx[_okBit]
            train_cx = np.setdiff1d(vm.train_cx, _test_cx)
        if len(filt['exclude_cids']) > 0:
            exclu_cx = cm.cx2_cid[filt['exclude_cids']]
            train_cx = np.setdiff1d(train_cx, exclu_cx)
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

    def num_train(vm):
        return len(vm.train_cid)
    def get_train_cx(vm):
        return vm.hs.cm.cid2_cx[vm.get_train_cid()]
    def get_test_cx(vm):
        return np.setdiff1d(vm.hs.cm.all_cxs(), vm.get_train_cx())
    def get_test_cid(vm):
        return vm.hs.cm.cx2_cid[vm.get_test_cx()]
    def get_train_cid(vm):
        return vm.train_cid
    def flip_sample(vm):
        vm.train_cid = vm.get_test_cid()
    def get_samp_id(vm):
        '''
        Returns an id unique to the sampled train_cid
        Note: if a cid is assigned to another.hip, this will break
        '''
        iom = vm.hs.iom
        samp_shelf = shelve.open(iom.get_temp_fpath('sample_shelf.db'))
        samp_key = '%r' % vm.train_cid
        if not samp_key in samp_shelf.keys():
            samp_shelf[samp_key] = len(samp_shelf.keys())+1
        samp_id = samp_shelf[samp_key]
        samp_shelf.close()
        return samp_id
    def get_samp_suffix(vm):
        return '.samp'+str(vm.get_samp_id())
