from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[training]')
import numpy as np
from hscom import helpers as util
from hscom import params
from hotspotter import chip_compute2 as cc2
from os.path import join


def get_training_output_dir(hs):
    output_dir = join(hs.dirs.computed_dir, 'training_examples')
    return output_dir


def vdg(hs):
    output_dir = get_training_output_dir(hs)
    util.vd(output_dir)


def generate_detector_training_data(hs, uniform_size=(512, 256)):
    print('')
    print('===================')
    print('Generating training data')
    lazy = util.get_flag('--lazy', True)
    output_dir = get_training_output_dir(hs)

    batch_extract_kwargs = {
        'lazy': lazy,
        'num_procs': params.args.num_procs,
        'force_gray': False,
        'uniform_size': uniform_size,
    }
    extract_detector_positives(hs, output_dir, batch_extract_kwargs)
    extract_detector_negatives(hs, output_dir, batch_extract_kwargs)


def extract_detector_negatives(hs, output_dir, batch_extract_kwargs):
    from itertools import product as iprod
    negreg_dir = join(output_dir, 'negatives', 'regions')
    negall_dir = join(output_dir, 'negatives', 'whole')
    negreg_fmt = join(negreg_dir, 'gx%d_wix%d_hix%d_neg.png')
    negall_fmt = join(negall_dir, 'gx%d_all_neg.png')
    util.ensuredir(negall_dir)
    util.ensuredir(negreg_dir)

    print('[train] extract_negatives')
    gx_list = hs.get_valid_gxs()
    nChips_list = np.array(hs.gx2_nChips(gx_list))
    aif_list = np.array(hs.gx2_aif(gx_list))

    # Find images where there are completely negative. They have no animals.
    #is_negative = np.logical_and(aif_list, nChips_list)
    is_completely_negative = np.logical_and(aif_list, nChips_list == 0)
    negall_gxs = gx_list[np.where(is_completely_negative)[0]]

    gfpath_list = []
    cfpath_list = []
    roi_list = []

    def add_neg_eg(roi, gfpath, cfpath):
        roi_list.append(roi)
        gfpath_list.append(gfpath)
        cfpath_list.append(cfpath)

    width_split = 2
    (uw, uh) = batch_extract_kwargs['uniform_size']

    for gx in negall_gxs:
        gfpath = hs.gx2_gname(gx, full=True)
        # Add whole negative image
        (gw, gh) = hs.gx2_image_size(gx)
        roi = (0, 0, gw, gh)
        add_neg_eg(roi, gfpath, negall_fmt % (gx))
        # Add negative regions
        w_step = gw // width_split
        h_step = int(round(gh * (w_step / gw)))
        nHeights, nWidths  = gh // h_step, gw // w_step
        if nWidths < 2 or nHeights < 1:
            continue
        for wix, hix in iprod(xrange(nWidths), xrange(nHeights)):
            x, y = wix * w_step, hix * h_step
            w, h = w_step, h_step
            roi = (x, y, w, h)
            add_neg_eg(roi, gfpath, negreg_fmt % (gx, wix, hix))

    theta_list = [0] * len(roi_list)

    cc2.batch_extract_chips(gfpath_list, cfpath_list, roi_list, theta_list,
                            **batch_extract_kwargs)


def extract_detector_positives(hs, output_dir, batch_extract_kwargs):
    print('[train] extract_positives')
    cx_list    = hs.get_valid_cxs()
    gx_list    = hs.tables.cx2_gx[cx_list]
    cid_list   = hs.tables.cx2_cid[cx_list]
    theta_list = hs.tables.cx2_theta[cx_list]
    roi_list   = hs.tables.cx2_roi[cx_list]
    gfpath_list = hs.gx2_gname(gx_list, full=True)

    posoutput_dir = join(output_dir, 'positives')
    util.ensuredir(posoutput_dir)
    pos_fmt = join(posoutput_dir, 'cid%d_gx%d_pos.png')
    cfpath_list = [pos_fmt  % (cid, gx) for (cid, gx) in zip(cid_list, gx_list)]

    cc2.batch_extract_chips(gfpath_list, cfpath_list, roi_list, theta_list,
                            **batch_extract_kwargs)

'''
python generate_training.py --dbdir /media/Store/data/work/MISC_Jan12
'''
if __name__ == '__main__':
    from hotspotter import main
    hs = main.main(defaultdb='MISC_Jan12', default_load_all=False)
    generate_detector_training_data(hs)
