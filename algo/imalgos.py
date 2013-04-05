from skimage.util.dtype import dtype_range
from skimage import exposure
from skimage.morphology import disk
from numpy import asarray, percentile, uint8, uint16

'''
raw_chip = cm.cx2_raw_chip(4)
img = raw_chip
figure(5)
pil_raw = Image.fromarray( raw_chip ).convert('L')
pil_filt = am.resize_chip(pil_raw, am.preproc.sqrt_num_pxls)
img = asarray(pil_filt)
'''

def contrast_stretch(img):
    # Contrast stretching
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return im_rescale

def histeq(img):
    'Local histogram equalization'
    # Equalization
    img_eq = exposure.equalize_hist(img)
    return img_eq

def adapt_histeq(img):
    # input uint8, output uint16
    img_adapteq = exposure.equalize_adapthist(uint16(img)*2**8, clip_limit=0.03) / uint16(2)**8
    return uint8(img_adapteq)
