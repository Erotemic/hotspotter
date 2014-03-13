from hsapi.HotSpotterAPI import HotSpotterAPI
from hsapi.util import ensure_path, assert_path, copy_all, copy, img_ext_set, copy_task
from fnmatch import fnmatch
from os import walk
from os.path import join, relpath, splitext
from PIL import Image

input_dir = '/media/Store/data/work/zebra_with_mothers'
output_dir = '/media/Store/data/work/HSDB_zebra_with_mothers'
output_img_dir = '/media/Store/data/work/HSDB_zebra_with_mothers/images'

convert_fmt = 'zebra_with_mothers'

assert_path(input_dir)
ensure_path(output_dir)
ensure_path(output_img_dir)

# Parses the zebra_with_mothers format
# into a hotspotter image directory
# which is logically named
cp_list = []
name_list = []
for root, dirs, files in walk(input_dir):
    foal_name = relpath(root, input_dir)
    for fname in files:
        chip_id, ext = splitext(fname)
        if not ext.lower() in img_ext_set:
            continue
        if chip_id.find('mother') > -1:
            chip_id = chip_id.replace('mother ', 'mom-')
            mom_name = chip_id.replace('mom-','') 
            name = mom_name
        else:
            name = foal_name
        dst_fname = 'Nid-' + name + '--Cid-' + chip_id + ext
        src = join(root, fname)
        dst = join(output_img_dir, dst_fname)
        name_list.append(name)
        cp_list.append((src, dst))
copy_task(cp_list, test=False, nooverwrite=True, print_tasks=True)

img_list   = [tup[1] for tup in cp_list]
sz_list    = [Image.open(_img).size for _img in img_list]
roi_list   = [(0,0,w,h) for (w,h) in sz_list]
theta_list = [0]*len(roi_list)

print([len(_) for _ in [img_list, sz_list, roi_list, theta_list]])

#hs = HotSpotterAPI(output_dir)
nx_list = hs.add_name_list2(name_list)
gx_list = hs.add_img_list2(img_list)
cx_list = hs.add_chip_list2(nx_list, gx_list, roi_list, theta_list)
hs.save_database()

if convert_fmt == 'zebra_with_mothers':
    img_list = create_bare_db_zebra_mother()

