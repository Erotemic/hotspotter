from __future__ import division
import load_data2
from hotspotter.helpers import ensure_path
from hotspotter.Parallelize import parallelize_tasks
from PIL import Image
import numpy as np

from hotspotter.algo.imalgos import histeq
def compute_chip(img_fpath, chip_fpath, roi, new_size):
    # Read image
    img = Image.open(img_fpath)
    [img_w, img_h] = [ gdim - 1 for gdim in img.size ]
    # Ensure ROI is within bounds
    [roi_x, roi_y, roi_w, roi_h] = [ max(0, cdim) for cdim in roi]
    roi_x2 = min(img_w, roi_x + roi_w)
    roi_y2 = min(img_h, roi_y + roi_h)
    # Crop out ROI: left, upper, right, lower
    raw_chip = img.crop((roi_x, roi_y, roi_x2, roi_y2))
    # Scale chip, but do not rotate
    pil_chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Preprocessing based on preferences
    pil_chip = histeq(pil_chip)
    # Save chip to disk
    pil_chip.save(chip_fpath, 'PNG')
    return True

# <Support for windows>
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    # </Support for windows>

    db_dir = load_data2.MOTHERS
    hs_tables = load_data2.load_csv_tables(db_dir)
    exec(hs_tables.execstr('hs_tables'))
    print(hs_tables)

    img_dir  = db_dir + '/images'
    chip_dir = db_dir + '/.hs_internals/computed/chips'
    feat_dir = db_dir + '/.hs_internals/computed/feats'

    ensure_path(chip_dir)
    ensure_path(feat_dir)

    cx2_chip_fpath = [chip_dir+'/CID_%d.png' % cid for cid in cx2_cid]
    cx2_img_fpath  = [ img_dir+'/'+gx2_gname[gx]   for gx  in cx2_gx ]
    # Compute the size of the new chip with a normalized area
    At = 20000.0 # target area
    def _target_resize(w, h):
        ht = np.sqrt(At * h / w)
        wt = w * ht / h
        return (int(round(wt)), int(round(ht)))
    cx2_chip_size = [_target_resize(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    cx2_imgchip_size = [(float(w), float(h)) for (x,y,w,h) in cx2_roi]

    #cx2_chip_sf = [(w/wt, h/ht) for ((w,h),(wt,ht)) in zip(cx2_imgchip_size, cx2_chip_size)]

    compute_chip_args = zip(cx2_img_fpath,
                            cx2_chip_fpath, 
                            cx2_roi,
                            cx2_chip_size)
    compute_chip_tasks = [(compute_chip, _args) for _args in compute_chip_args]

    parallelize_tasks(compute_chip_tasks, num_procs=8)
