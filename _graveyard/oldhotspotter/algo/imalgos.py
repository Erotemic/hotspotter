from numpy import asarray, percentile, uint8, uint16
import numpy as np
from hotspotter.other.logger import logdbg, logwarn
from PIL import Image, ImageOps

'''
from skimage.util.dtype import dtype_range
from skimage import exposure
from skimage.morphology import disk
from numpy import asarray, percentile, uint8, uint16
from matplotlib.pyplot import *
cm  = hs.cm
fig = figure(42)
raw_chip = cm.cx2_raw_chip(4)
img = raw_chip
pil_raw = Image.fromarray( raw_chip ).convert('L')
pil_filt = am.resize_chip(pil_raw, am.preproc.sqrt_num_pxls)
img = asarray(pil_filt)
imshow(img)
fig.show()
'''
# http://scikit-image.org/docs/dev/api/skimage.filter.html#denoise-bilateral
# How can I hold all of these algorithms?!

def contrast_stretch(img):
    from skimage.morphology import disk
    from skimage.util.dtype import dtype_range
    from skimage import exposure
    # Contrast stretching
    p2 = percentile(img, 2)
    p98 = percentile(img, 98)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    print img_rescale.dtype
    return img_rescale

def histeq(pil_img):
    img = asarray(pil_img)
    try:
        from skimage import exposure
        'Local histogram equalization'
        # Equalization
        img_eq_float64 = exposure.equalize_hist(img)
        return Image.fromarray(uint8(np.round(img_eq_float64*255)))
    except Exception as ex:
        from hotspotter.tpl.other import imtools
        logdbg('Scikits not found: %s' % str(ex))
        logdbg('Using fallback histeq')
        return Image.fromarray(imtools.histeq(img)).convert('L')

def adapt_histeq(img):
    try: 
        from skimage import exposure
        # input uint8, output uint16
        img_uint8  = img
        img_uint16 = uint16(img)*2**8
        img_adapteq_uint16 = exposure.equalize_adapthist(img_uint16,\
                                                        ntiles_x=8,\
                                                        ntiles_y=8,\
                                                        clip_limit=0.01,\
                                                        nbins=256)
        img_adapteq_cropped_uint8 = uint8(img_adapteq_uint16[5:-5][5:-5] / uint16(2)**8 )
        return img_adapteq_cropped_uint8
    except Exception as ex:
        logdbg('Scikits not found: %s' % str(ex))
        logwarn('Scikits not found: %s' % str(ex))
        return img


def bilateral_filter(img):
    try:
        import skimage.filter
        img_uint8 = img
        img_float = float32(img_uint8) / 255
        #mode = Points outside the boundaries of the input are filled according to the given mode (?constant?, ?nearest?, ?reflect? or ?wrap?). Default is ?constant?.
        img_bilat = skimage.filter.denoise_bilateral(img_float, win_size=20, sigma_range=1.6, sigma_spatial=1.6, bins=256, mode='reflect', cval='reflect')
        return img_bilat
    except Exception as ex:
        logdbg('Scikits not found: %s' % str(ex))
        logwarn('Scikits not found: %s' % str(ex))
        return img

def segment(img):
    raise NotImplementedError
    import scimage.segmentation.felzenswalb

def superpixels(img):
    raise NotImplementedError
    import scimage.segmentation.slic

def skelatonize(img):
    raise NotImplementedError
    import scimage.morphology.skeletonize
