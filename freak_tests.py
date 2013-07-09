import matplotlib
print('Configuring matplotlib for Qt4')
matplotlib.use('Qt4Agg')

import hotspotter.tpl.cv2 as cv2


from pylab import *
from hotspotter.helpers import Timer, figure, myprint
from hotspotter.other.ConcretePrintable import Pref
import os, sys, types, string


# ___HELPER_FUNCTIONS___
# --------
def read_img(chip_fpath):
    '''
    Reads an image into a numpy array. 
    Formated as gray scale.
    Checks existence
    Resizes to 100x100
    '''
    if not os.path.exists(chip_fpath):
        print('Chip fpath DOES NOT EXIST!: '+chip_fpath)
        return None
    # Read Image
    img3 = cv2.imread(chip_fpath)
    img2  = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img   = cv2.resize(img2,(100,100))
    return img
# --------
def black_bar(img):
    '''
    Places a vertical black bar in a numpy or opencv image
    '''
    img2 = np.copy(img)
    mid  = img2.shape[1]/2
    hw   = 2
    img2[:, (mid-hw):(mid+hw)] = 0
    return img2
# --------
cv_flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
def show_image(img, cv_kpts=None, fignum=0, title=None):
    print('Drawing Image')
    figure(fignum=fignum, doclf=True, title=title)
    if not cv_kpts is None:
        img = cv2.drawKeypoints(img, cv_kpts, flags=cv_flags)
    imshow(img)
    set_cmap('gray')
# --------
def get_opencv_params(opencv_class):
    opencv_pref = Pref()
    for param_name in opencv_class.getParams():
        param_type = opencv_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            param_val = opencv_class.getInt(param_name)
        elif param_type == 1:
            param_val = opencv_class.getBool(param_name)
        elif param_type in [2,7]:
            param_val = opencv_class.getDouble(param_name)
        else:
            raise Exception('Unknown opencv param. name: '+str(param_name) + ' type: '+str(param_type))
        opencv_pref[param_name] = param_val
        return opencv_pref

def set_opencv_params(opencv_class, param_dict):
    for param_name, param_val in param_dict.iteritems():
        param_type = opencv_class.paramType(param_name)
        if param_type in [0, 9, 11]:
            opencv_class.setInt(param_name, param_val)
        elif param_type == 1:
            opencv_class.setBool(param_name, param_val)
        elif param_type in [2,7]:
            opencv_class.setDouble(param_name, param_val)
        else:
            raise Exception('Unknown opencv param. name: '+str(param_name) + ' type: '+str(param_type))
        opencv_pref[param_name] = param_val
# --------
def detect_and_extract(img,
                       kpts_type='SIFT',
                       desc_type='SIFT',
                       gravity=True,
                       cv_params=None):
    '''
    Detects and extracts keypoints with opencv
    '''
    print('Detect and Extract')
    # Define Detector
    if type(kpts_type) == types.StringType:
        kpts_detector  = cv2.FeatureDetector_create(kpts_type)
        #kpts_detector = cv2.PyramidAdaptedFeatureDetector(kpts_detector3, maxLevel=1)
        #kpts_detector = cv2.GridAdaptedFeatureDetector(kpts_detector3) # max number of features
        with Timer(msg='Detecting %s keypoints' % kpts_type):
            cv_kpts2 = kpts_detector.detect(img)  
        if gravity:
            for cv_kp in iter(cv_kpts2): cv_kp.angle = 0
    else:
        cv_kpts2 = kpts_type
    # Define Extactor
    extractor = cv2.DescriptorExtractor_create(desc_type)
    if not cv_params is None:
        for key, val in cv_params.iteritems():
            # TODO: This is not always gaurenteed to be a bool
            extractor.setBool(key, val)
    with Timer(msg='Extracting %d %s descriptors' % (len(cv_kpts2), desc_type)):
        cv_kpts, cv_descs = extractor.compute(img, cv_kpts2)
    print('+%s has %d keypoints\n' % (desc_type, len(cv_kpts)))
    return cv_kpts, cv_descs
# --------
def get_all_figures():
    all_figures=[manager.canvas.figure for manager in
                 matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    return all_figures
def show_all_figures():
    for fig in iter(get_all_figures()):
        fig.show()
        fig.canvas.draw()
def move_all_figures():
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        #myprint(fig.canvas.manager.window) 
        # the manager should be a qt window
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        num_rows = 3
        h = 300
        w = 300
        y = (i%num_rows)*w
        x = (int(i/num_rows))*h
        qtwin.setGeometry(x,y,w,h)
def bring_to_front_all_figures():
    from PyQt4.QtCore import Qt
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        qtwin.raise_()
        qtwin.activateWindow()
        qtwin.setWindowFlags(Qt.WindowStaysOnTopHint)
        qtwin.show()
        qtwin.setWindowFlags(Qt.WindowFlags(0))
        qtwin.show()
        #what is difference between show and show normal?
def close_all_figures():
    from PyQt4.QtCore import Qt
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        qtwin.close()
# --------

# ___DEFINE_PARAMETERS___

# Define Image
if sys.platform == 'win32':
    chip_fpath = 'D:/data/work/PZ_flankhack/images/img-0000001.jpg'
else: 
    chip_fpath =   '/media/Store/data/work/zebra_with_mothers/06_410/1.JPG'

# OpenCV Feature Detector Benchmarks
# http://computer-vision-talks.com/2011/01/comparison-of-the-opencvs-feature-detection-algorithms-2/
# Conclusion: FAST is fast, STAR has low error

# OpenCV Feature Detector Documentation
# http://docs.opencv.org/modules/features2d/doc/feature_detection_and_description.html#freak-freak

import load_data2
db_dir = load_data2.NAUTS
hs_tables = load_data2.load_csv_tables(db_dir)
exec(hs_tables.execstr('hs_tables'))
print(hs_tables)


kpts_type_pref = Pref('SIFT', choices=['SIFT', 'SURF', 'ORB', 'BRISK', 'BRIEF'])
kpts_type = kpts_type_pref.value()

__SIFT_PARAMS__ = ['contrastThreshold', 'edgeThreshold', 'nFeatures', 'nOctaveLayers', 'sigma']
__FREAK_PARAMS__ = ['nbOctave', 'orientationNormalized', 'patternScale', 'scaleNormalized']

img = read_img(chip_fpath)
#in_place_black_bar(img)

if False:
    sift_cv_kpts, sift_cv_desc =\
            detect_and_extract(img, kpts_type=kpts_type, desc_type='SIFT')
    freak_cv_kpts, freak_cv_desc =\
            detect_and_extract(img, kpts_type=kpts_type, desc_type='FREAK',
                            cv_params={'orientationNormalized':False})
    show_image(img,  sift_cv_kpts, fignum=1,
            title='%d SIFT features' % len(sift_cv_kpts))
    show_image(img, freak_cv_kpts, fignum=2,
            title='%d FREAK features' % len(freak_cv_kpts))
    show_all_figures()

__SINGLE_KEYPOINT_EXPERIMENT__ = True
if __SINGLE_KEYPOINT_EXPERIMENT__:
    # 
    mykpt = cv2.KeyPoint(50, 50, _size=10, _angle=0, _response=1, _octave=1)
    mysift_cv_kpts, mysift_cv_desc =\
            detect_and_extract(img, kpts_type=[mykpt], desc_type='SIFT')
    myfreak_cv_kpts, myfreak_cv_desc =\
            detect_and_extract(img, kpts_type=[mykpt], desc_type='FREAK',
                               cv_params={'orientationNormalized':False})

    img2 = black_bar(img)
    mysift_cv_kpts2, mysift_cv_desc2 =\
            detect_and_extract(img2, kpts_type=[mykpt], desc_type='SIFT')
    myfreak_cv_kpts2, myfreak_cv_desc2 =\
            detect_and_extract(img2, kpts_type=[mykpt], desc_type='FREAK',
                               cv_params={'orientationNormalized':False})
    show_image(img,  mysift_cv_kpts, fignum=1, title='One SIFT feature')
    show_image(img, myfreak_cv_kpts, fignum=2, title='One FREAK feature')
    show_image(img2, myfreak_cv_kpts, fignum=3, title='One Occluded FREAK feature')

    def cv_freak_to_binary(myfreak_cv_desc):
        getBin = lambda x, n: x >= 0 and str(bin(x))[2:].zfill(n) or "-" + str(bin(x))[3:].zfill(n)
        freak_binary_list = []
        for freak_desc in myfreak_cv_desc:
            bin_freak_byte_list = []
            for freak_byte in freak_desc:
                bin_byte = getBin(freak_byte, 8)
                #print ('%3d ' % freak_byte )+ str(bin_byte)
                bin_freak_byte = [b=='1' for b in bin_byte]
                bin_freak_byte_list.append(bin_freak_byte)
            bin_freak_desc = np.hstack(bin_freak_byte_list)
            freak_binary_list.append(bin_freak_desc)
        freak_binary = np.vstack(freak_binary_list)
        return freak_binary

    freak_binary = cv_freak_to_binary(myfreak_cv_desc)
    freak_binary2 = cv_freak_to_binary(myfreak_cv_desc2)
    
    # Find where the FREAK descriptor deviates
    myfreak = freak_binary[0]
    myfreak2 = freak_binary2[0]
    diff_dims = []
    for i, (f1, f2) in enumerate(zip(freak_binary[0], freak_binary2[0])):
        #print('Dim '+str(i)+': '+str(int(f1))+' '+str(int(f2)))
        if f1 != f2:
            diff_dims.append(i)
    num_diff        = len(diff_dims)
    num_fine_diff   = (np.array(diff_dims) > 478).sum()
    num_coarse_diff = (np.array(diff_dims) < 16).sum()

    mysift = mysift_cv_desc[0]
    mysift2 = mysift_cv_desc2[0]
    # Distances between blackbar and normal SIFT descriptor
    l2_sift_dist  = np.sqrt(np.sum((mysift - mysift2)**2))
    l1_sift_dist  = np.sum(np.abs(mysift - mysift2))
    # Energy of SIFT descriptor (should always be constant)
    l1_energy = np.sum(mysift)
    l1_energy2 = np.sum(mysift2)
    l2_energy = np.sqrt(np.sum(mysift**2))
    l2_energy2 = np.sqrt(np.sum(mysift2**2))
    print('Occlusion Difference: ')
    print('  FREAK All   : '+str(num_diff))+'/512'
    print('  FREAK Fine  : '+str(num_fine_diff))+'/32'
    print('  FREAK Coarse: '+str(num_coarse_diff))+'/16'
    print('  ------------')
    print('  SIFT L2dist : '+str(l2_sift_dist)+\
          '  ( / '+str(l1_energy)+' / '+str(l1_energy2)+' / )')
    print('  SIFT L1dist : '+str(l1_sift_dist)+\
          '  ( / '+str(l2_energy)+' / '+str(l2_energy2)+' / )')

    # FIGURE 5
    figure(5, title='Different Freak Dimensions', doclf=True)
    hist(diff_dims, bins=512, color=[1,0,0])
    ax = gca()
    ax.set_xlim(0,512)
    # FIGURE 6
    figure(6, title='Freak Descriptor Histograms', doclf=True)
    hist(np.nonzero(myfreak), bins=512, alpha=.3, color=[0,1,0])
    hist(np.nonzero(myfreak2), bins=512, alpha=.3, color=[0,0,1])
    ax = gca()
    ax.set_xlim(0,512)

    show_all_figures()
    move_all_figures()

try:
    __IPYTHON__
    show_all_figures()
    bring_to_front_all_figures()
except NameError as ex:
    if ex.message != '''name '__IPYTHON__' is not defined''':
        raise
    else:
        if '--cmd' in sys.argv:
            from hotspotter.helpers import in_IPython, have_IPython
            run_exec = False
            if not in_IPython() and have_IPython():
                import IPython
                IPython.embed()
        else:
            show()
