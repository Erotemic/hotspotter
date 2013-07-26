import os, sys, string
import fnmatch
import numpy as np
from hotspotter.helpers import checkpath, unit_test, ensure_path, symlink, remove_files_in_dir
from hotspotter.helpers import myprint
from hotspotter.other.ConcretePrintable import DynStruct

def printDBG(msg, lbl=''):
    print('DBG: '+lbl+str(msg))

rdir_img      = '/images'
rdir_internal = '/.hs_internals'
rdir_chip     = rdir_internal + '/computed/chips'

class HotspotterTables(DynStruct):
    def __init__(self):
        super(HotspotterTables, self).__init__()
        self.gx2_gname    = []
        self.nx2_name     = []
        self.cx2_cid      = []
        self.cx2_nx       = []
        self.cx2_gx       = []
        self.cx2_roi      = []
        self.cx2_theta    = []
        self.px2_propname = []
        self.px2_cx2_prop = []

class HotspotterDirs(DynStruct):
    def __init__(self, db_dir):
        super(HotspotterDirs, self).__init__()
        internal_dir = db_dir + '/.hs_internals'
        # Class variables
        self.db_dir       = db_dir
        self.img_dir      = db_dir + rdir_img
        self.internal_sym = db_dir + '/Shortcut-to-hs_internals'
        self.internal_dir = db_dir + rdir_internal
        self.chip_dir     = db_dir + rdir_chip
        self.rchip_dir    = internal_dir + '/computed/temp'
        self.feat_dir     = internal_dir + '/computed/feats'
        # Make directories if needbe
        ensure_path(self.internal_dir)
        ensure_path(self.chip_dir)
        ensure_path(self.rchip_dir)
        ensure_path(self.feat_dir)
        if not os.path.islink(self.internal_sym):
            symlink(internal_dir, self.internal_sym, noraise=True)

    def delete_computed_dir(self):
        computed_dir = self.internal_dir + '/computed'
        remove_files_in_dir(computed_dir, recursive=True)

def tryindex(list, val):
    try: 
        return list.index(val)
    except ValueError as ex:
        return -1

def load_csv_tables(db_dir):
    '''
    Big function which loads the csv tables from a datatabase directory
    Returns HotspotterDirs and HotspotterTables
    '''
    print('\n=============================')
    print('Loading hotspotter csv tables: '+str(db_dir))
    print('=============================')
    hs_dirs = HotspotterDirs(db_dir)
    #exec(hs_dirs.execstr('hs_dirs'))
    #print(hs_dirs.execstr('hs_dirs'))
    feat_dir     = hs_dirs.feat_dir
    img_dir      = hs_dirs.img_dir
    internal_sym = hs_dirs.internal_sym
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir
    internal_dir = hs_dirs.internal_dir
    db_dir       = hs_dirs.db_dir
    # --- Table Names ---
    chip_table   = internal_dir + '/chip_table.csv'
    name_table   = internal_dir + '/name_table.csv'
    image_table  = internal_dir + '/image_table.csv' # TODO: Make optional
    # --- CHECKS ---
    hasDbDir   = checkpath(db_dir)
    hasImgDir  = checkpath(img_dir)
    hasChipTbl = checkpath(chip_table)
    hasNameTbl = checkpath(name_table)
    hasImgTbl  = checkpath(image_table)
    if not all([hasDbDir, hasImgDir, hasChipTbl, hasNameTbl, hasImgTbl]):
        errmsg = ''
        errmsg+=('\n\n!!!!!\n\n')
        errmsg+=('  ! The datatables seem to not be loaded')
        errmsg+=(' Files in internal dir: '+repr(internal_dir))
        for fname in os.listdir(internal_dir):
            errmsg+=('   ! fname') 
        errmsg+=('\n\n!!!!!\n\n')
        print(errmsg)
        raise Exception(errmsg)
    print('-------------------------')
    print('Loading database tables: ')
    cid_lines  = [] 
    line_num   = 0
    csv_line   = ''
    csv_fields = []
    try:
        # ------------------
        # --- READ NAMES --- 
        # ------------------
        print('... Loading name table: '+name_table)
        nx2_name = ['____', '____']
        nid2_nx  = { 0:0, 1:1}
        name_lines = open(name_table,'r')
        for line_num, csv_line in enumerate(name_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            nid = int(csv_fields[0])
            name = csv_fields[1]
            nid2_nx[nid] = len(nx2_name)
            nx2_name.append(name)
        print('      * Loaded '+str(len(nx2_name)-2)+' names (excluding unknown names)')
        print('      * Done loading name table')

        # -------------------
        # --- READ IMAGES --- 
        # -------------------
        gx2_gname = []
        print('... Loading images')
        # Load Image Table 
        # <LEGACY CODE>
        print('    ... Loading image table: '+image_table)
        gid2_gx = {}
        gid_lines = open(image_table,'r').readlines()
        for line_num, csv_line in enumerate(gid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            gid = int(csv_fields[0])
            if len(csv_fields) == 3: 
                gname = csv_fields[1]
            if len(csv_fields) == 4: 
                gname = csv_fields[1:3]
            gid2_gx[gid] = len(gx2_gname)
            gx2_gname.append(gname)
        nTableImgs = len(gx2_gname)
        fromTableNames = set(gx2_gname)
        print('          * table specified '+str(nTableImgs)+' images')
        # </LEGACY CODE>
        # Load Image Directory
        print('    ... Loading image directory: '+img_dir)
        nDirImgs = 0
        nDirImgsAlready = 0
        for fname in os.listdir(img_dir):
            if len(fname) > 4 and string.lower(fname[-4:]) in ['.jpg', '.png', '.tiff']:
                if fname in fromTableNames: 
                    nDirImgsAlready += 1
                    continue
                gx2_gname.append(fname)
                nDirImgs += 1
        print('          * dir specified '+str(nDirImgs)+' images')
        print('          * '+str(nDirImgsAlready)+' were already specified in the table')
        print('  * Loaded '+str(len(gx2_gname))+' images')
        print('  * Done loading images')

        # ------------------
        # --- READ CHIPS --- 
        # ------------------
        print('... Loading chip table: '+chip_table)
        # Load Chip Table Header
        cid_lines = open(chip_table,'r').readlines()
        # Header Markers
        header_numdata   = '# NumData '
        header_csvformat = '# ChipID,'
        # Default Header Variables
        chip_csv_format = ['ChipID', 'ImgID',  'NameID',   'roi[tl_x  tl_y  w  h]',  'theta']
        num_data   = -1
        # Parse Chip Table Header
        for line_num, csv_line in enumerate(cid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0:
                continue
            csv_line = csv_line.strip('\n')
            if csv_line.find('#') != 0:
                break # Break after header
            if csv_line.find(header_csvformat) == 0:
                chip_csv_format = [_.strip() for _ in csv_line.strip('#').split(',')]
            if csv_line.find(header_numdata) == 0:
                num_data = int(csv_line.replace(header_numdata,''))
        print('  * num_chips: '+str(num_data))
        print('  * chip_csv_format: '+str(chip_csv_format))
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
        # Hotspotter Chip Tables
        cx2_cid   = []
        cx2_nx    = []
        cx2_gx    = []
        cx2_roi   = []
        cx2_theta = []
        # x is a csv field index in this context
        px2_propname = [chip_csv_format[x] for x in prop_x_list]
        px2_cx2_prop = [[] for px in prop_x_list]
        print('  * num_user_properties: '+str(len(prop_x_list)))
        # Parse Chip Table
        for line_num, csv_line in enumerate(cid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
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
    except Exception as ex:
        print('Failed parsing: '+str(''.join(cid_lines)))
        print('Failed on line number:  '+str(line_num))
        print('Failed on line:         '+repr(csv_line))
        print('Failed on fields:       '+repr(csv_fields))
        raise

    print('  * Loaded: '+str(len(cx2_cid))+' chips')
    print('  * Done loading chip table')
    # Return all information from load_tables
    hs_tables = HotspotterTables()
    #hs_tables.gid2_gx = gid2_gx
    #hs_tables.nid2_nx  = nid2_nx
    hs_tables.gx2_gname  = np.array(gx2_gname)
    hs_tables.nx2_name   = np.array(nx2_name)
    hs_tables.cx2_cid      = np.array(cx2_cid)
    hs_tables.cx2_nx       = np.array(cx2_nx)
    hs_tables.cx2_gx       = np.array(cx2_gx)
    hs_tables.cx2_roi      = np.array(cx2_roi)
    hs_tables.cx2_theta    = np.array(cx2_theta)
    hs_tables.px2_propname = np.array(px2_propname)
    hs_tables.px2_cx2_prop = np.array(px2_cx2_prop)
    print('===============================')
    print('Done Loading hotspotter csv tables: '+str(db_dir))
    print('===============================\n\n')
    return hs_dirs, hs_tables

    

def __print_chiptableX(hs_tables):
    #print(hs_tables.execstr('hs_tables'))
    #exec(hs_tables.execstr('hs_tables'))
    cx2_gx    = hs_tables.cx2_gx
    cx2_cid   = hs_tables.cx2_cid
    cx2_nx    = hs_tables.cx2_nx
    cx2_theta = hs_tables.cx2_theta
    cx2_roi   = hs_tables.cx2_roi
    #prop_names = ','.join(px2_propname)
    print('=======================================================')
    print('# Begin ChipTableX')
    print('# ChipID, NameX,  ImgX,     roi[tl_x  tl_y  w  h],  theta')
    chip_iter = iter(zip(cx2_cid, cx2_nx, cx2_gx, cx2_roi, cx2_theta))
    for (cid, nx, gx, roi, theta) in chip_iter:
        print('%8d, %5d, %5d, %25s, %6.3f' % (cid, nx, gx, str(roi).replace(',',''), theta))
    print('# End ChipTableX')
    print('=======================================================')

def print_chiptable(hs_tables):
    #exec(hs_tables.execstr('hs_tables'))
    #print(hs_tables.execstr('hs_tables'))
    #prop_names = ','.join(px2_propname)
    print('=======================================================')
    print('# Begin ChipTable')
    # Get length of the max vals for formating
    cx2_cid   = hs_tables.cx2_cid
    cx2_theta = hs_tables.cx2_theta
    cx2_gname = [hs_tables.gx2_gname[gx] for gx in  hs_tables.cx2_gx]
    cx2_name  = [hs_tables.nx2_name[nx]  for nx in  hs_tables.cx2_nx]
    cx2_stroi = [str(roi).replace(',','') for roi in  hs_tables.cx2_roi]
    max_gname = max([len(gname) for gname in iter( cx2_gname)])
    max_name  = max([len(name)  for name  in iter( cx2_name) ])
    max_stroi = max([len(stroi) for stroi in iter( cx2_stroi)])
    _mxG = str(max([max_gname+1, 5]))
    _mxN = str(max([max_name+1, 4]))
    _mxR = str(max([max_stroi+1, 21]))

    fmt_str = '%8d, %'+_mxN+'s, %'+_mxG+'s, %'+_mxR+'s, %6.3f'

    c_head = '# ChipID'
    n_head = ('%'+_mxN+'s') %  'Name'
    g_head = ('%'+_mxG+'s') %  'Image'
    r_head = ('%'+_mxR+'s') %  'roi[tl_x  tl_y  w  h]'
    t_head = ' theta'
    header = ', '.join([c_head,n_head,g_head,r_head,t_head])
    print(header)

    # Build the table
    chip_iter = iter(zip( cx2_cid, cx2_name, cx2_gname, cx2_stroi, cx2_theta))
    for (cid, name, gname, stroi, theta) in chip_iter:
        _roi  = str(roi).replace(',',' ') 
        print(fmt_str % (cid, name, gname, stroi, theta))

    print('# End ChipTable')
    print('=======================================================')


# MODULE GLOBAL VARIABLES
WORK_DIR = 'D:/data/work'
if sys.platform == 'linux2':
    WORK_DIR = '/media/Store/data/work'
# Common databases I use
FROGS   = WORK_DIR+'/FROG_tufts'
NAUTS   = WORK_DIR+'/NAUT_Dan'
GZ_ALL  = WORK_DIR+'/GZ_ALL'
WS_HARD = WORK_DIR+'/WS_hard'
MOTHERS = WORK_DIR+'/HSDB_zebra_with_mothers'


@unit_test
def test_load_csv():
    db_dir = MOTHERS
    hs_dirs, hs_tables = load_csv_tables(db_dir)
    print_chiptable(hs_tables)
    __print_chiptableX(hs_tables)
    print(hs_tables.nx2_name)
    print(hs_tables.gx2_gname)
    hs_tables.printme2(val_bit=True, max_valstr=10)
    return hs_dirs, hs_tables

# Test load csv tables
if __name__ == '__main__':
    from load_data2 import *
    hs_dirs, hs_tables = test_load_csv()
    if '--cmd' in sys.argv:
        from hotspotter.helpers import in_IPython, have_IPython
        run_exec = False
        if not in_IPython() and have_IPython():
            import IPython
            IPython.embed()
