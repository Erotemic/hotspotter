import os
import types
import subprocess
import shelve
import numpy as np
from hotspotter.other.logger  import logmsg, logdbg, logerr, logwarn
from hotspotter.other.ConcretePrintable import DynStruct, Pref 
from hotspotter.other.AbstractPrintable import AbstractManager
import cv2
from PIL import Image, ImageOps
import re


class AlgorithmManager(AbstractManager): 
    '''Manager the settings for different algorithms
    as well as the implementation of some and calling of other.
    spatial_functions (which contains ransac) should be consolidated
    here.'''
    def __init__(am, hs):
        super( AlgorithmManager, am ).__init__( hs)
        am.default_depends = ['preproc','chiprep','model','query', 'results']
        am.algo_prefs = None
        am.last_algo_key = None
        am.init_preferences()

    def init_preferences(am, default_bit=False):
        iom = am.hs.iom
        if am.algo_prefs == None:
            am.algo_prefs = Pref(fpath=iom.get_prefs_fpath('algo_prefs'))
            #Define the pipeline stages
            am.algo_prefs.preproc  = Pref(parent=am.algo_prefs)  # Low Level Chip Operations
            am.algo_prefs.chiprep  = Pref(parent=am.algo_prefs)  # Extracting Chip Features
            am.algo_prefs.model    = Pref(parent=am.algo_prefs, hidden=True)  # Building the model
            am.algo_prefs.query    = Pref(parent=am.algo_prefs)  # Searching the model
            am.algo_prefs.results  = Pref(parent=am.algo_prefs)  # Searching the model
        # --- Chip Preprocessing ---
        # (selection, options, prefs? )
        am.algo_prefs.preproc.sqrt_num_pxls           = Pref(700)
        am.algo_prefs.preproc.autocontrast_bit        = Pref(False,hidden=True)
        am.algo_prefs.preproc.bilateral_filt_bit      = Pref(False)
        am.algo_prefs.preproc.histeq_bit              = Pref(True)
        am.algo_prefs.preproc.contrast_stretch_bit    = Pref(False)
        am.algo_prefs.preproc.adapt_histeq_bit        = Pref(False)
        # --- Chip Representation ---
        # Currently one feature detector and one feature descriptor is chosen
        am.algo_prefs.chiprep.use_gravity_vector     = True

        opencv_detectors = ['FAST', 'STAR', 'SIFT', 'SURF', 'ORB', 'BRISK',
                            'MSER', 'GFTT', 'HARRIS', 'Dense', 'SimpleBlob']
        # Keypoint Detector: Add a list of opencv detectors
        am.algo_prefs.chiprep.kpts_detector           = Pref(0, choices=['heshesaff']+opencv_detectors) 
        # Keypoint Detector Parameters: Add a list of associated preferences
        # types = { 0:'int', 1:'bool', 2:'double', 7:'float', 9:'int64', 11:'unsigned char'}
        for detector_type in opencv_detectors:
            det_dep = (am.algo_prefs.chiprep.kpts_detector_internal, detector_type)
            det_pref = Pref(depeq=det_dep)
            det = cv2.FeatureDetector_create(detector_type)
            for param_name in det.getParams():
                param_type = det.paramType(param_name)
                if param_type in [0, 8, 9, 11]:
                    param_val = det.getInt(param_name)
                elif param_type == 1:
                    param_val = det.getBool(param_name)
                elif param_type in [2,7]:
                    param_val = det.getDouble(param_name)
                else:
                    raise Exception('name: '+str(param_name) + ' type: '+str(param_type))
                det_pref[param_name] = param_val
            am.algo_prefs.chiprep[detector_type+'_params'] = det_pref

        am.algo_prefs.chiprep.kpts_extractor          = Pref(0, choices=('SIFT',), hidden=True) #, '#SURF', '#BRISK'))
        # --- Vocabulary ---
        am.algo_prefs.model.quantizer                 = Pref(0, choices=('naive_bayes', 'akmeans'), hidden=True)

        #nbnn_dep = (am.algo_prefs.model.quantizer_internal, 'naive_bayes')
        #am.algo_prefs.model.naive_bayes                    = Pref(depeq=nbnn_dep)
        #am.algo_prefs.model.naive_bayes.num_nearest        = Pref('wip')
        #am.algo_prefs.model.naive_bayes.pseudo_num_words   = Pref('wip')

        akm_dep = (am.algo_prefs.model.quantizer_internal, 'akmeans')
        am.algo_prefs.model.akmeans             = Pref(depeq=akm_dep, hidden=True)
        am.algo_prefs.model.akmeans.num_words   = Pref(1000)
        am.algo_prefs.model.akmeans.max_iters   = Pref(1000)

        hkm_dep = (am.algo_prefs.model.quantizer_internal, 'hkmeans')
        am.algo_prefs.model.hkmeans             = Pref(depeq=hkm_dep, hidden=True)
        am.algo_prefs.model.hkmeans.branching   = Pref(10)
        am.algo_prefs.model.hkmeans.depth       = Pref(6)
        
        flann_kdtree = Pref(hidden=True)
        flann_kdtree.algorithm  = Pref(default=1, choices=['linear',
                                                 'kdtree',
                                                 'kmeanes',
                                                 'composite',
                                                 'autotuned']) # Build Prefs
        flann_kdtree.trees      = Pref(8, min=0, max=30)
        flann_kdtree.checks     = Pref(128, min=0, max=4096) # Search Prefs
        #Autotuned Specific Prefeters
        autotune_spef = (flann_kdtree.algorithm_internal, 'autotuned') 
        flann_kdtree.target_precision = Pref(0.95, depeq=autotune_spef)  
        flann_kdtree.build_weight     = Pref(0.01, depeq=autotune_spef) 
        flann_kdtree.memory_weight    = Pref(0.86, depeq=autotune_spef,\
                                             doc='the time-search tradeoff') 
        flann_kdtree.sample_fraction  = Pref(0.86, depeq=autotune_spef,\
                                             doc='the train_fraction')
        # HKMeans Specific Prefeters
        hkmeans_spef = (flann_kdtree.algorithm_internal, 'kmeans') #Autotuned Specific Prefeters
        flann_kdtree.branching    = Pref(10, depeq=hkmeans_spef) 
        flann_kdtree.iterations   = Pref( 6, depeq=hkmeans_spef, doc='num levels') 
        flann_kdtree.centers_init = Pref(choices=['random', 'gonzales', 'kmeansapp'],\
                                         depeq=hkmeans_spef) 
        flann_kdtree.cb_index = Pref(0, min=0, max=5, depeq=hkmeans_spef, doc='''
            this parameter (cluster boundary index) influences the way exploration
            is performed in the hierarchical kmeans tree. When cb index is
            zero the next kmeans domain to be explored is choosen to be the one with
            the closest center. A value greater then zero also takes into account the
            size of the domain.''' ) 
        am.algo_prefs.model.indexer = flann_kdtree #Pref(0, choices=[flann_kdtree])

        # --- Query Prefs ---
        am.algo_prefs.query.k                         = Pref(1,    min=1, max=50)
        am.algo_prefs.query.num_rerank                = Pref(1000, min=0)
        am.algo_prefs.query.spatial_thresh            = Pref(0.05, min=0, max=1) 
        am.algo_prefs.query.sigma_thresh              = Pref(0.05, min=0, max=1, hidden=True) #: Unimplemented
        am.algo_prefs.query.method                    = Pref(2, choices=['COUNT', 'DIFF', 'LNRAT', 'RAT']) #, '#TFIDF'
        am.algo_prefs.query.score                     = Pref(0, choices=['cscore','nscore']) # move to results?
        am.algo_prefs.query.self_as_result_bit        = Pref(False)  #: Return self (in terms of name) in results
        am.algo_prefs.query.remove_other_names        = Pref(False)  #: Remove all results with the same identified name as the query

        # --- Result Prefs --- 
        am.algo_prefs.results.score                   = Pref(0, choices=('cscore','nscore')) # move to results?
        am.algo_prefs.results.one_result_per_name     = Pref(False)  #: Return self (in terms of name) in results
        am.algo_prefs.results.match_threshold         = Pref(0) 
        am.algo_prefs.results.min_num_results         = Pref(5)
        am.algo_prefs.results.max_num_results         = Pref(5) 
        am.algo_prefs.results.extra_num_results       = Pref(0)
        if not default_bit:
            am.algo_prefs.load()

    def get_algo_name(am, depends, abbrev_bit=False):
        if depends == 'all':
            depends = am.default_depends
        if not abbrev_bit:
            # No abreviation = nice printable representation
            exclude_list = []
            for stage in am.default_depends:
                if stage not in depends:
                    exclude_list.append(stage)
            print_attri = am.algo_prefs.get_printable(type_bit=False,print_exclude_aug=exclude_list)
            return 'Algorithm Prefs'+('\n'+print_attri).replace('\n','\n    ')
        abbrev_list = [
            ('use_gravity_vector','gv'),
            ('histeq_bit','hsteq'),
            ('num_top','nTop'),
            ('kpts_extractor','dsc'),
            ('kpts_detector','fp'),
            ('autocontrast_bit','autocontr'),
            ('bilateral_filt_bit','bilat'),
            ('sqrt_num_pxls','sz'),
            ('truncate','trunk'),
            ('True','T'),
            ('False','F')
        ]
        stage_list = []
        for stage_name in depends:
            stage_struct = am.algo_prefs[stage_name] # Get the dependent stage
            stage_par = stage_name+' {'+stage_struct.get_printable(type_bit=False)
            stage_par = stage_par.replace('\n',',') #remove newlines
            stage_par = re.sub('[\' \"]','',stage_par) #remove extra characters
            for abbrev_tup in abbrev_list: # replace abbreviations
                stage_par = stage_par.replace(abbrev_tup[0],abbrev_tup[1])
            stage_par = re.sub('\,$','',stage_par)
            stage_list.append(stage_par+'}')
        return '_'.join(stage_list)

    def get_algo_id(am, depends):
        'Returns an id unique to the current algorithm'
        iom = am.hs.iom
        algo_shelf = shelve.open(iom.get_temp_fpath('algo_shelf.db'))
        algo_key = am.get_algo_name(depends=depends,abbrev_bit=True)
        shelf_changed_bit = False
        if not algo_key in algo_shelf.keys():
            algo_shelf[algo_key] = len(algo_shelf.keys())+1
            shelf_changed_bit = True
        algo_id = algo_shelf[algo_key]
        algo_shelf.close()
        if shelf_changed_bit:
            am.write_algo_id_table()
        return algo_id

    def write_algo_id_table(am):
        'write the algo_id shelf to human readable format'
        iom = am.hs.iom
        to_write = 'Algorithm ID Table\n'
        algo_shelf = shelve.open(iom.get_temp_fpath('algo_shelf.db'))
        for algo_key in algo_shelf.keys():
            # write backwards because keys here are big
            to_write += str(algo_shelf[algo_key]) + ' <: ' + algo_key + '\n'
        algo_shelf.close()
        with open(iom.get_temp_fpath('algo_shelf_table.txt'),'w') as f:
            print to_write
            f.write(to_write)


    def get_algo_suffix(am, depends, use_id_bit=True):
        if type(depends) == types.ListType and len(depends) == 0:
            return ''

        if use_id_bit:
            return '.algo.'+str(am.get_algo_id(depends))
        else:
            return '.algo.'+am.get_algo_name(depends,True)

    # DEPRICATED
    def preprocess_chip(am, pil_filt):
        logdbg('prepocessing')
        # Convert to grayscale
        # raw_chip = cm.cx2_raw_chip(6)
        # --- Resize ---

        # --- Filters ---
        #if am.algo_prefs.preproc.histeq_bit : 
            ##pil_filt = ImageOps.equalize(pil_filt)
            #img_rescale = exposure.equalize_hist(np.asarray(pil_filt))
            #pil_filt = Image.fromarray(histeq(np.asarray(pil_filt))).convert('L')
        if am.algo_prefs.preproc.histeq_bit:
            from hotspotter.algo.imalgos import histeq
            logdbg('Equalizing Histogram')
            pil_filt = histeq(pil_filt)
        if am.algo_prefs.preproc.adapt_histeq_bit:
            from hotspotter.algo.imalgos import adapt_histeq
            logdbg('Adaptive Equalizing Histogram')
            pil_filt = Image.fromarray(adapt_histeq(np.asarray(pil_filt)))
        if am.algo_prefs.preproc.contrast_stretch_bit:
            from hotspotter.algo.imalgos import contrast_stretch
            logdbg('Stretching Histogram')
            pil_filt = Image.fromarray(contrast_stretch(np.asarray(pil_filt)))
        if am.algo_prefs.preproc.autocontrast_bit :
            logdbg('PIL Autocontrast')
            pil_filt = ImageOps.autocontrast(pil_filt)
        if am.algo_prefs.preproc.bilateral_filt_bit :
            logdbg('O(1) Bilateral Filter Approximation')
            from hotspotter.tpl.other.shiftableBF import shiftableBF
            pil_filt = Image.fromarray(shiftableBF(np.asarray(pil_filt)))

        return pil_filt

    # DEPRICATED   
    def compute_vlfeatures(am, chip):
        import vlfeat

    # DEPRICATED
    def compute_features(am, chip):
        'Computes features of a chip. Uses settings in AlgorithmManager'
        logdbg('Calling feature detector')
        external_detectors = ['heshesaff', 'heslapaff', 'heslap', 'harlap', 'dense']
        external_descriptors = ['SIFT']

        if am.algo_prefs.chiprep.kpts_detector in external_detectors:
            (kpts, desc) = am.external_feature_computers(chip)
            if am.algo_prefs.chiprep.kpts_extractor in external_descriptors:
                return (kpts, desc)
        else:
            logerr('Only External Keypoint Detectors are implemented: '+str(external_detectors))
        logerr('Only External Keypoint Descriptors are implemented: '+str(external_descriptors))

        # http://stackoverflow.com/questions/10764895/opencv-python-sample-error
        #  http://stackoverflow.com/questions/12491022/opencv-freak-fast-retina-keypoint-descriptor
        # The following detector types are supported: 
            # FAST, STAR, SIFT (nonfree), SURF (nonfree),
            # ORB, BRISK, MSER, GFTT (good features to track)
            # HARRIS, Dense, SimpleBlob
            # Also: Grid, GridFAST, PyramidStar
            # see http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html#Ptr<FeatureDetector> FeatureDetector::create(const string& detectorType)
        im  = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        logdbg('Making detector: %r' % am.algo_prefs.chiprep.kpts_detector)
        cvFeatDetector  = cv2.FeatureDetector_create(am.algo_prefs.chiprep.kpts_extractor)
        #cvFeatDetector   = cv2.PyramidAdaptedFeatureDetector(cvFeatDetector_,4)
        logdbg('Made %r, Making extractor: %r' % (cvFeatDetector, am.algo_prefs.chiprep.kpts_detector))
        cvFeatExtractor  = cv2.DescriptorExtractor_create(am.algo_prefs.chiprep.kpts_extractor)

        logdbg('Made %r, Detecting keypoints on image' % cvFeatExtractor )
        cvKpts_= cvFeatDetector.detect(im)
        # Tinker with Keypoint
        if am.algo_prefs.chiprep['use_gravity_vector']: 
            for cvKp in cvKpts_:
                cvKp.angle = 0
                r = cvKp.size #scale = (r**2)/27
        logdbg('Made %r, Keypoint description with %d kpts ' % (cvFeatExtractor, len(cvKpts_)) )
        (cvKpts, cvDesc) = cvFeatExtractor.compute(im,cvKpts_)
        logdbg('Detected %d features  ' % len(cvKpts) )
        kpts = np.zeros((len(cvKpts), 5),dtype=np.float32)
        desc = np.array(cvDesc,dtype=np.uint8)
        fx = 0
        # * Convert to representation in: M. Perdoc, O. Chum, and J. Matas. CVPR 2009
        # * Efficient representation of local geometry for large scale object retrieval
        for cvKP in cvKpts:
            (x,y) = cvKP.pt
            theta = cvKP.angle
            scale = (float(cvKp.size)**2)/27
            detA  = 1./(scale)
            (a,c,d) = (detA, 0, detA)
            kpts[fx] = (x,y,a,c,d)
            fx += 1
        return (kpts, desc)
         # Garbage
         # SIFT descriptors are computed with a radius of r=3*np.sqrt(3*s)
         # s = (det A_i) ^ (-1/2) OR
         # s = sqrtm(inv(det(A_i)))
         #aIS = 1/np.sqrt(a)
         #cIS = (c/np.sqrt(a) - c/np.sqrt(d))/(a - d + eps)
         #dIS = 1/np.sqrt(d)
         #print (aIS,cIS,dIS)
         #print (a,c,d)
         #print '_-----------'
         #    if False:
         #        hessian_threshold = 85
         #        surfDetector_   = cv2.SURF(hessianThreshold)
         #        surfDetector_   = cv2.GridAdaptedFeatureDetector(surfDetector_,50)
         #        surfDetector    = cv2.PyramidAdaptedFeatureDetector(surfDetector_,50)
         #        freakExtractor  = cv2.DescriptorExtractor_create('FREAK')

    # DEPRICATED
    def  external_feature_computers(am, chip):
        'Write chip ; call extern executable ; read output ; return (kpts,desc)'
        logdbg('Calling external kpt detector')
        iom = am.hs.iom
        chip = Image.fromarray(chip)
        tmp_chip_fpath = iom.get_temp_fpath('tmp.ppm')
        chip.save(tmp_chip_fpath,'PPM')
        perdoch_external = ['heshesaff']
        mikolajczyk_external = ['heslapaff','dense']
        if am.algo_prefs.chiprep.kpts_detector in perdoch_external:
            exename = iom.get_hesaff_exec()
            outname = tmp_chip_fpath+'.hesaff.sift'
            args = '"'+tmp_chip_fpath+'"'
        elif am.algo_prefs.chiprep.kpts_detector in mikolajczyk_external:
            exename = iom.get_inria_exec()
            feature_name = am.algo_prefs.chiprep.kpts_detector
            if feature_name == 'heslapaff':
                feature_name = 'hesaff'
                suffix = 'hesaff'
            if feature_name == 'dense':
                feature_name = feature_name+' 6 6'
                suffix = 'dense'
            outname = tmp_chip_fpath+'.'+suffix+'.sift'
            args = '-'+feature_name+' -sift -i "'+tmp_chip_fpath+'"'
        else:
            logerr('Method %r + %r is invalid in extern_detect_kpts.m'\
                   % (am.algo_prefs.chiprep.kpts_detector, am.algo_prefs.chiprep.kpts_extractor))

        cmd = exename+' '+args
        logdbg('External Executing: %r ' % cmd)
        try:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logdbg('External Execution did not throw an error')
            (out, err) = proc.communicate()
            logdbg(str(out)+' '+str(err))
            if proc.returncode != 0:
                logerr('Failed to execute '+cmd+'\n OUTPUT: '+out)
            if not os.path.exists(outname):
                logerr('The output file doesnt exist: '+outname)
            logdbg('External Output:\n'+out[:-1])
        except Exception as ex:
            logwarn('An Exception occurred while calling the keypoint detector: '+str(ex))
            try:
                ret2 = os.system(cmd)
                if ret2 != 0:
                    logerr(str(ex)+'\nThe backup keypoint detector didnt work either!')
            except Exception as ex2:
                logerr(str(ex2))
        fid = file(outname,'r')
        ndims  = int(fid.readline())
        nkpts = int(fid.readline())
        if ndims != 128: 
            raise Exception(' These are not SIFT dexcriptors ')
        kpts = np.zeros((nkpts,5),  dtype=np.float32)
        desc = np.zeros((nkpts,ndims),dtype=np.uint8)
        lines = fid.readlines()
        # SIFT descriptors are computed with a radius of r=3*np.sqrt(3*s)
        # s = (det A_i) ^ (-1/2) OR
        # s = sqrtm(inv(det(A_i)))

        for i in range(nkpts):
            nums = lines[i].split(' ')
            kpts[i,:] = np.array(map(lambda _: float(_)   , nums[0:5]),dtype=np.float32)
            desc[i,:] = np.array(map(lambda _: np.uint8(_), nums[5:]),dtype=np.uint8)
        fid.close()

        return (kpts, desc)
        '''
            Interest point detectors/descriptors implemented by K.Mikolajczyk@surrey.ac.uk
            at [ref. http://www.robots.ox.ac.uk/~vgg/research/affine]
            Options:
            Interest points:
                -harris - harris detector
                -hessian - hessian detector
                -harmulti - multi-scale harris detector
                -hesmulti - multi-scale hessian detector
                -harhesmulti - multi-scale harris-hessian detector
                -harlap - harris-laplace detector
                -heslap - hessian-laplace detector
                -dog    - DoG detector
                -mser   - mser detector
                -haraff - harris-affine detector
                -hesaff - hessian-affine detector
                -harhes - harris-hessian-laplace detector
                -dense dx dy - dense sampling
            Interest points prefeters:
                -density 100 - feature density per pixels (1 descriptor per 100pix)
                -harThres - harris threshold [100]
                -hesThres  - hessian threshold [200]
                -edgeLThres  - lower canny threshold [5]
                -edgeHThres  - higher canny threshold [10]
            Descriptors:
                -sift - sift [D. Lowe]
                -gloh - gloh [KM]
            Descriptor prefenters:
                -color - color sift [KM]
                -dradius - patch radius for computing descriptors at scale 1
                -fface ..../facemodel.dat - frontal face detector
            Input/Output:
                -i image.png  - input image pgm, ppm, png, jpg, tif
                -p1 image.pgm.points - input regions format 1
                -p2 image.pgm.points - input regions format 2
                -o1 out.desc - saves descriptors in out.desc output format 1
                -o2 out.desc - saves descriptors in out.desc output format 2
                -noangle - computes rotation variant descriptors (no rotation esimation)
                -DP - draws features as points in out.desc.png
                -DC - draws regions as circles in out.desc.png
                -DE - draws regions as ellipses in out.desc.png
                -c 255 - draws points in grayvalue [0,...,255]'''

