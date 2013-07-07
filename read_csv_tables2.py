import os
import sys
import numpy as np

WORK_DIR = 'D:/data/work'
if sys.platform == 'linux2':
    WORK_DIR = '/media/Store/data/work'
'''
os.listdir(WORK_DIR)
'''

FROGS = WORK_DIR+'/FROG_tufts'
NAUTS = WORK_DIR+'/NAUT_DAN'
GZ_ALL = WORK_DIR+'/GZ_ALL'

TEMP = WORK_DIR+'/TheRopes'


# SELECT A DATABASE
DB_DIR = GZ_ALL

image_dir  = DB_DIR+'/images'
chip_table = DB_DIR+'/.hs_internals/chip_table.csv'
name_table = DB_DIR+'/.hs_internals/name_table.csv'
# TODO: Make optional
image_table = DB_DIR+'/.hs_internals/image_table.csv'


# --- READ NAMES --- #
print('Loading name table: '+name_table)
nx2_name = ['____', '____']
nid2_nx  = { 0:0, 1:1}
name_lines = open(name_table,'r')
for csv_line in iter(name_lines):
    if csv_line.find('#') == 0:
        continue
    csv_fields = [_.strip(' ') for _ in csv_line.strip('\n').split(',')]
    nid = int(csv_fields[0])
    name = csv_fields[1]
    nid2_nx[nid] = len(nx2_name)
    nx2_name.append(name)
print('  * Loaded '+str(len(nx2_name)-2)+' names (excluding unknown names)')
print('  * Done loading name table')

# --- READ IMAGES --- #
gx2_gname = []
# Load Image Table 
# <LEGACY CODE>
print('Loading image table: '+image_table)
gid2_gx = {}
gid_lines = open(image_table,'r').readlines()
for csv_line in iter(gid_lines):
    if csv_line.find('#') == 0:
        continue
    csv_fields = [_.strip(' ') for _ in csv_line.strip('\n').split(',')]
    gid = int(csv_fields[0])
    if len(csv_fields) == 3: 
        gname = csv_fields[1]
    if len(csv_fields) == 4: 
        gname = csv_fields[1:3]
    gid2_gx[gid] = len(gx2_gname)
    gx2_gname.append(gname)
# </LEGACY CODE>

# Load Image Directory
print('Loading image directory')
for fname in os.listdir(image_dir):
    if len(fname) > 4 and string.lower(fname[-4]) in ['.jpg', '.png', '.tiff']:
        gx2_gname.append(fname)
print('  * Loaded '+str(len(gx2_gname))+' images')
print('  * Done loading images')

# --- READ CHIPS --- #
print('Loading chip table: '+chip_table)
# Load Chip Table
cx2_gx = []
# Load Chip Table Header
cid_lines = open(chip_table,'r').readlines()
# Header Markers
header_numdata = '# NumData '
header_csvformat = '# ChipID,'
# Default Header Variables
chip_csv_format = ['ChipID', 'ImgID',  'NameID',   'roi[tl_x  tl_y  w  h]',  'theta']
num_data   = -1
for csv_line in iter(cid_lines):
    csv_line = csv_line.strip('\n')
    if csv_line.find('#') != 0:
        break
     # Parse Header
    if csv_line.find(header_csvformat) == 0:
        chip_csv_format = [_.strip() for _ in csv_line.strip('#').split(',')]
    if csv_line.find(header_numdata) == 0:
        num_data = int(csv_line.replace(header_numdata,''))

print('  * num_chips: '+str(num_data))
print('  * chip_csv_format: '+str(chip_csv_format))

def tryindex(list, val):
    try: 
        return list.index(val)
    except ValueError as ex:
        return -1

cid_x   = tryindex(chip_csv_format, 'ChipID')
gid_x   = tryindex(chip_csv_format, 'ImgID')
nid_x   = tryindex(chip_csv_format, 'NameID')
roi_x   = tryindex(chip_csv_format, 'roi[tl_x  tl_y  w  h]')
theta_x = tryindex(chip_csv_format, 'theta')
# new fields
gname_x = tryindex(chip_csv_format, 'Image')
name_x  = tryindex(chip_csv_format, 'Name')

required_x = [cid_x, gid_x, gname_x, nid_x, name_x, roi_x, theta_x]
prop_x_list = np.setdiff1d(range(len(chip_csv_format)), required_x).tolist()

cx2_cid   = []
cx2_nx    = []
cx2_gx    = []
cx2_roi   = []
cx2_theta = []
# x is a csv field index in this context
px2_propname = [chip_csv_format[x] for x in prop_x_list]
px2_cx2_prop = [[] for px in prop_x_list]

print('  * num_user_properties: '+str(len(prop_x_list)))

for csv_line in iter(cid_lines):
    if csv_line.find('#') == 0:
        continue
    csv_fields = [_.strip(' ') for _ in csv_line.strip('\n').split(',')]
    # Load Chip ID
    cid = int(csv_fields[cid_x])
    # Load Chip Image Info
    if gid_x != -1:
        gid = int(csv_fields[gid_x])
        gx = gid2_gx[gid]
    elif gname_x != -1:
        gname = csv_fields[gname_x]
        gx = gx2_name.index(gname)
    # Load Chip Name Info
    if nid_x != -1:
        nid = int(csv_fields[nid_x])
        nx = nid2_nx[nid]
    elif name_x != -1:
        name = csv_fields[name_x]
        nx = nx2_name.index(name)
    # Load Chip ROI Info
    roi_str = csv_fields[roi_x].strip('[').strip(']')
    roi = [int(_) for _ in roi_str.split()]
    # Load Chip theta Info
    if theta_x != -1:
        theta = float(csv_fields[theta_x])
    else:
        theta = 0
    # Append info to cid lists
    cx2_cid.append(cid)
    cx2_gx.append(gx)
    cx2_nx.append(nx)
    cx2_roi.append(roi)
    cx2_theta.append(theta)
    for px, x in enumerate(prop_x_list): 
        px2_cx2_prop[px].append(csv_fields[x])

print('  * Loaded: '+str(len(cx2_cid))+' chips')
print("  * Finished reading chip table")


def print_chiptable():
    prop_names = ','.join(px2_propname)
    print('ChipID, NameX,  ImgX,     roi[tl_x  tl_y  w  h],  theta')
    chip_iter = iter(zip(cx2_cid, cx2_nx, cx2_gx, cx2_roi, cx2_theta))
    for (cid, nx, gx, roi, theta) in chip_iter:
        print('%6d, %5d, %5d, %25s, %6.3f' % (cid, nx, gx, roi, theta))

    print('\n\nChipID,             Name,            Image,    roi[tl_x  tl_y  w  h],  theta')
    chip_iter = iter(zip(cx2_cid, cx2_nx, cx2_gx, cx2_roi, cx2_theta))
    for (cid, nx, gx, roi, theta) in chip_iter:
        print('%6d, %16s, %16s, %24s, %6.3f' % (cid, nx2_name[nx], gx2_gname[gx], roi, theta))

print_chiptable()
