from __future__ import division, print_function
import numpy as np
from hotspotter import helpers
from hotspotter import chip_compute2 as cc2
from os.path import join

uniform_size = (512, 256)


def generate_detector_training_data(hs):
    print('')
    print('===================')
    print('Generating training data')
    lazy = helpers.argv_flag('--lazy', True)
    output_dir = join(hs.dirs.computed_dir, 'training_examples')
    batch_extract_kwargs = {
        'lazy': lazy,
        'num_procs': hs.args.num_procs,
        'force_gray': False,
        'uniform_size': uniform_size,
    }
    #extract_detector_positives(hs, output_dir, batch_extract_kwargs)
    extract_detector_negatives(hs, output_dir, batch_extract_kwargs)


def extract_detector_negatives(hs, output_dir, batch_extract_kwargs):
    negoutput_dir = join(output_dir, 'negatives')
    helpers.ensuredir(negoutput_dir)
    neg_fmt = join(negoutput_dir, 'gx%d_wix%d_hix%d_neg.png')

    print('[train] extract_negatives')
    gx_list = hs.get_valid_gxs()
    nChips_list = np.array(hs.gx2_nChips(gx_list))
    aif_list = np.array(hs.gx2_aif(gx_list))

    # Find images where there are completely negative. They have no animals.
    #is_negative = np.logical_and(aif_list, nChips_list)
    is_completely_negative = np.logical_and(aif_list, nChips_list == 0)
    cneg_gxs = gx_list[np.where(is_completely_negative)[0]]

    gfpath_list = []
    cfpath_list = []
    roi_list = []

    width_split = 2

    (uw, uh) = uniform_size
    for gx in cneg_gxs:
        gfpath = hs.gx2_gname(gx, full=True)
        (gw, gh) = hs.gx2_image_size(gx)
        w_stride = gw // width_split
        h_stride = int(round(gh * (w_stride / gw)))
        num_heights = gh // h_stride
        num_widths = gw // w_stride
        if num_heights < 1 or num_widths < 2:
            continue
        for wix in xrange(num_widths):
            for hix in xrange(num_widths):
                x = wix * w_stride
                y = hix * h_stride
                w = w_stride
                h = h_stride
                roi = (x, y, w, h)
                roi_list += [roi]
                gfpath_list += [gfpath]
                cfpath_list += [neg_fmt % (gx, wix, hix)]

    theta_list = [0] * len(roi_list)

    if batch_extract_kwargs['lazy']:
        helpers.vd(negoutput_dir)

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
    helpers.ensuredir(posoutput_dir)
    pos_fmt = join(posoutput_dir, 'cid%d_gx%d_pos.png')
    cfpath_list = [pos_fmt  % (cid, gx)
                   for (cid, gx) in zip(cid_list, gx_list)]
    if batch_extract_kwargs['lazy']:
        helpers.vd(posoutput_dir)

    cc2.batch_extract_chips(gfpath_list, cfpath_list, roi_list, theta_list,
                            **batch_extract_kwargs)


if __name__ == '__main__':
    from hotspotter import main
    hs = main.main(defaultdb='MISC_Jan12', default_load_all=False)
    generate_detector_training_data(hs)
