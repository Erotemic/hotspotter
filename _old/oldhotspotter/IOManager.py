# IOManager
#
# Contains logic to read and write all current formats.
# Maintains directory structures and maintains
# all data that is stored on disk (more or less)
# it at least handles the names. Sometimes more.
#
# Name conventions: 
# fname - filename (no path)
# dname - directory (no path)
# fpath - full path to file
# dpath - full path to directory

import re
import shutil
import time
import fnmatch
import sys
import hotspotter.tpl
import numpy as np
import pylab
import os
from os.path import expanduser, join, relpath, realpath, normpath, exists, dirname
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.helpers import dircheck
from hotspotter.other.logger import logmsg, logwarn, logdbg, logerr, logio
from hotspotter.other.crossplat import platexec

#----------------
def checkdir_decorator(method_fn):
    def wrapper(iom, *args):
        ret = method_fn(iom, *args)
        dircheck(ret)
        return ret
    return wrapper

class IOManager(AbstractManager):
    
    def __init__(iom, hs):
        super( IOManager, iom ).__init__( hs )        
        logdbg('Creating IOManager')
        iom.hs = hs
        iom._hsroot = None
        iom.settings_dpath = normpath(join(expanduser('~'),'.hotspotter'))
        iom.internal_dname = '.hs_internals';
        iom.dummy_delete = False #Dont actually delete things
        iom.find_hotspotter_root_dir()

    def hsroot(iom):
        if iom._hsroot is None:
            iom.find_hotspotter_root_dir()
        return iom._hsroot

    def find_hotspotter_root_dir(iom):
        # Find the HotSpotter root dir even in installed packages
        hsroot = realpath(dirname(__file__))
        while True:
            root_landmark = join(hsroot, '__HOTSPOTTER_ROOT__')
            logdbg('Testing Existence:'+str(root_landmark))
            if not os.path.exists(root_landmark):
                logdbg('No landmark here')
            else: 
                logdbg('Found the landmark')
                break
            _newroot = dirname(hsroot)
            if _newroot == hsroot:
                logerr('Cannot Find HotSpotter Root')
            hsroot = _newroot
        iom._hsroot = hsroot

    def remove_file(iom, fpath):
        if iom.dummy_delete:
            logdbg('DummyDelete: %s' % fpath)
            return False
        logdbg('Deleting: %s' %  fpath)
        try:
            os.remove(fpath)
        except OSError as e:
            logwarn('OSError: %s,\n Could not delete %s' % (str(e), fpath))
            return False
        return True

    def remove_files_with_pattern(iom, dpath, fname_pattern, recursive_bit=True):
        logdbg('Removing files in directory %r %s' % (dpath, ['', ', Recursively'][recursive_bit]))
        logdbg('Removing files with pattern: %r' % fname_pattern)
        num_removed = 0
        num_matched = 0
        for root, dname_list, fname_list in os.walk(dpath):
            for fname in fnmatch.filter(fname_list, fname_pattern):
                num_matched += 1
                num_removed += iom.remove_file(join(root, fname))
            if not recursive_bit:
                break
        logmsg('Removed %d/%d files' % (num_removed, num_matched))
        return True


    def remove_settings_files_with_pattern(iom, fname_pattern):
        iom.remove_files_with_pattern(iom.settings_dpath, fname_pattern, recursive_bit=False)
        'removes files in computed_dpath'

    def remove_computed_files_with_pattern(iom, fname_pattern):
        iom.remove_files_with_pattern(iom.get_computed_dpath(), fname_pattern, recursive_bit=True)
        'removes files in computed_dpath'

    # DEPRICATED
    def get_tpl_lib_dir(iom):
        return join(dirname(hotspotter.tpl.__file__), 'lib', sys.platform)
     #START: Directory and File Managment
    #==========
    # --- Private Directories'
    @checkdir_decorator
    def  get_internal_dpath(iom):
        return join(iom.hs.db_dpath, iom.internal_dname)
    @checkdir_decorator
    def  get_computed_dpath(iom):
        return join(iom.get_internal_dpath(),'computed')
    @checkdir_decorator
    def  get_thumb_dpath(iom, thumb_type):
        return join(iom.get_computed_dpath(), 'thumbs', thumb_type)
    def  get_experiment_dpath(iom):
        return join(iom.get_computed_dpath(), 'experiments')
    # --- Public Directories
    @checkdir_decorator
    def  get_img_dpath(iom):
        return join(iom.hs.db_dpath, 'images')
    @checkdir_decorator
    def  get_chip_dpath(iom):
        return join(iom.get_computed_dpath(), 'chips')
    @checkdir_decorator
    def  get_chiprep_dpath(iom):
        return join(iom.get_computed_dpath(), 'features')
    @checkdir_decorator
    def  get_model_dpath(iom):
        return join(iom.get_computed_dpath(), 'models')
    @checkdir_decorator
    def  get_temp_dpath(iom):
        return join(iom.get_computed_dpath(), 'temp')
    def  get_temp_fpath(iom, tmp_fname):
        return normpath(join(iom.get_temp_dpath(), tmp_fname))
    def  get_user_fpath(iom, fname):
        return normpath(join(iom.hs.db_dpath, fname))
    def  write_to_user_fpath(iom, fname, to_write):
        user_fpath = iom.get_user_fpath(fname)
        iom.logwrite(user_fpath, to_write)

    # DEPRICATED: TODO: REMOVE
    def logwrite(iom, fpath, to_write):
        iom.write(fpath, to_write) 

    def get_timestamp(iom):
        'Year-Month-Day_Hour-Minute'
        import datetime
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    @checkdir_decorator
    def ensure_directory(iom, dpath):
        'Makes directory if it does not exist. Returns path to directory'
        return dpath

    @checkdir_decorator
    def ensure_computed_directory(iom, dname):
        'Input: Path relative to the computed directory'
        'Ensures directory exists in the database\'s computed directory.'
        'Output: Returns absolute path'
        return join(iom.get_computed_dpath(), dname)

    def write(iom, fpath, to_write):
        logmsg('Writing to: %s' % fpath)
        print 'Writing String:\n%s' % to_write
        try:
            with open(fpath, 'w') as f:
                    f.write(to_write)
            print 'Wrote to %s' % fpath
        except Exception as ex: 
            print 'Error: '+str(ex)
            print 'Failed to write to %s ' % fpath
    # --- Main Saved Files
    def  get_image_table_fpath(iom):
        return normpath(join(iom.get_internal_dpath(),'image_table.csv'))
    def  get_chip_table_fpath(iom):
        return normpath(join(iom.get_internal_dpath(),'chip_table.csv'))
    def  get_name_table_fpath(iom):
        return normpath(join(iom.get_internal_dpath(),'name_table.csv'))
    def  get_flat_table_fpath(iom):
        return normpath(join(iom.hs.db_dpath,'flat_table.csv'))
    # --- Executable Filenames
    def  get_hesaff_exec(iom):
        ext = ''
        if sys.platform == 'win32':
            ext = '.exe'
        tpl_dir = dirname(hotspotter.tpl.__file__)
        tpl_hesaff = normpath(join(tpl_dir, 'hesaff', 'hesaff'+ext))
        if os.path.exists(tpl_hesaff):
            return tpl_hesaff
        # Fix for weird mac packaging things
        root_dir = tpl_dir
        while root_dir!=None:
            tpl_hesaff = join(root_dir, 'hotspotter', 'tpl', 'hesaff', 'hesaff'+ext)
            logdbg(tpl_hesaff)
            exists_test = os.path.exists(tpl_hesaff)
            logdbg('Exists:'+str(exists_test))
            if exists_test:
                break
            tmp = os.path.dirname(root_dir)
            if tmp == root_dir:
                root_dir = None
            else:
                root_dir = tmp
        return '"' + tpl_hesaff + '"'

    def  get_inria_exec(iom):
        return platexec(join(iom.get_tpl_lib_dir(), 'inria_features'))
    # --- Chip Representations
    def get_chip_prefix(iom, cid, depends):
        'Naming convention for chips: cid, algo_depends, other' 
        am = iom.hs.am
        algo_suffix = am.get_algo_suffix(depends=depends)
        return 'CID.'+str(cid)+'_'+algo_suffix
    def  get_chiprep_fpath(iom, cid):
        chiprep_fname = iom.get_chip_prefix\
                (cid, ['preproc', 'chiprep']) + '_feats.npz' 
        return normpath(join(iom.get_chiprep_dpath(), chiprep_fname))
    # Images thumb and full
    def  get_img_thumb_fpath(iom, gname):
        return normpath(join(iom.get_thumb_dpath('images'), gname))
    def  get_img_fpath(iom, gname):
        return normpath(join(iom.get_img_dpath(), gname))
    # Chips thumb and full
    def  get_chip_thumb_fpath(iom, cid):
        chip_fname = iom.get_chip_prefix(cid, ['preproc'])+'_chip.jpg' 
        return normpath(join(iom.get_thumb_dpath('chip'), chip_fname))
    def  get_chip_fpath(iom, cid):
        chip_fname = iom.get_chip_prefix(cid, ['preproc'])+'_chip.png' 
        return normpath(join(iom.get_chip_dpath(),chip_fname))
    # 
    def get_model_fpath(iom):
        am, vm = iom.hs.get_managers('am','vm')
        algo_suffix = am.get_algo_suffix(depends=['preproc','chiprep','model'])
        samp_suffix = vm.get_samp_suffix()
        model_fname = 'model'+samp_suffix+algo_suffix+'.npz'
        return normpath(join(iom.get_model_dpath(),model_fname))
    def get_flann_index_fpath(iom):
        am, vm = iom.hs.get_managers('am','vm')
        algo_suffix = am.get_algo_suffix(['preproc','chiprep','model'])
        samp_suffix = vm.get_samp_suffix()
        flann_index_fname = 'index%s%s.flann' % (algo_suffix, samp_suffix)
        return normpath(join(iom.get_model_dpath(), flann_index_fname))
    # --- Indexes 
    def get_prefs_fpath(iom, prefs_name):
        dircheck(iom.settings_dpath)
        return normpath(join(iom.settings_dpath,  prefs_name+'.txt'))

    def get_dataset_fpath(iom, db_name=None):
        if sys.platform == 'win32':
            work_fpath = 'D:/data/work/'
        else:
            work_fpath = '/data/work/'
        if db_name is None:
            db_name = 'Naut_Dan'
            print "Valid Work Directories Are: "
            for dir in os.listdir(work_fpath):
                print dir
        return work_fpath+db_name

    def  load_tables(iom):
        logmsg('Loading data tables in '+iom.hs.db_dpath)
        if not (exists(iom.get_image_table_fpath()) and\
                exists(iom.get_name_table_fpath()) and\
                exists(iom.get_image_table_fpath())):
            if exists(iom.get_oxford_gt_dpath()):
                logmsg('You have selected an Oxford style groundtruth')
                iom.load_oxford_gt()
                logmsg('Succesfully Loaded Oxford style groundtruth')
                sys.stdout.flush()
                return
            logwarn('Trying to load a Legacy Database')
        iom.load_image_table()
        iom.load_name_table()
        iom.load_chip_table()
        logmsg('Done loading data tables')
        sys.stdout.flush()

    #START: CSV IO METHODS

    #=======================================
    # IO Internals
    def  _load_table(iom, csv_fpath, table_name, alloc_func, csv_func):
        '''
        Reads csv files. Must pass in a table name a memory allocation function 
        and a csv_func: function which parses the fields read by _load_table
        '''
        logio('Loading '+table_name+' Table: '+csv_fpath)
        if not exists(csv_fpath):
            logio('\"'+csv_fpath+'\" Does Not Exist')
            return False
        fid = file(csv_fpath, 'r')
        csv_headers = None
        line = fid.readline()
        num_line_prefix = '# NumData'
        # Foreach line in the CSV file
        while line != '':
            line = line.strip()
            # NEW LINE: Skip
            if line == '': continue
            # COMMENT LINE: Check for metadata
            elif line[0] == '#':
                # CHECK Preallocation
                if line.find(num_line_prefix) > -1:
                    # Parse out the number of lines to allocate
                    # and use the given allocation function
                    num_lines = int(line.replace(num_line_prefix,'').replace(' ',''))
                    alloc_func(num_lines)
                # CHECK Data Headers: StripeSpotter
                elif line.find('#imgindex') > -1:
                    logmsg('Loading a Legacy StripeSpotter File')
                    csv_headers = line[1:].split(',')
                # CHECK Data Headers: Legacy HotSpotter 
                elif line.find('#01)') > -1:
                    logmsg('Loading a Legacy HotSpotter File')
                    csv_headers = []
                    while line != '':
                        line = line[:-1]
                        if len(line) < 4 or line[3] != ')': break
                        parnstr = '#\\d\\d\\) '
                        head_field = re.sub(parnstr, '', line)
                        head_field = re.sub(' - .*','', head_field)
                        csv_headers += [head_field]
                        line = fid.readline()
                # CHECK Data Headers: Hotspotter
                elif any([line.find(field) >=0 for field in ['ChipID', 'NameID', 'ImageID']]):
                    csv_headers = [field.strip() for field in line[1:].split(',')]
                    # HACK: Change the fields to the ones it actually expects
                    import hotspotter.other.AbstractPrintable
                    _lbl2_header = hotspotter.other.AbstractPrintable._lbl2_header
                    _header2_lbl = {v:k for k,v in _lbl2_header.iteritems()}
                    csv_headers = [_header2_lbl[field] if field in _header2_lbl.keys() else field for field in csv_headers]
                    
            # DATA LINE: Read it
            else:
                csv_data = [data_field.strip() for data_field in line.split(',')]
                csv_func(csv_data, csv_headers)
            # Next Line
            line = fid.readline()
        # Finsh reading table
        fid.close()
        logio('Loaded '+table_name+' Table')
        return True

    def __image_csv_func(iom, csv_data, csv_headers=None):
        """ A function which reads a single line of csv image data """
        '''
        gid   = None
        gname = None
        aif   = None
        if csv_headers != None: pass'''
        if len(csv_data) == 3:
            # Format where extension is part of name
            gid   = int(csv_data[0])
            gname = csv_data[1]
            aif   = csv_data[2]
            logdbg('Adding Image')
        elif len(csv_data) == 4:
            # Format where extension is its own field
            gid   = int(csv_data[0])
            gnameext    = csv_data[2]
            gname_noext = csv_data[1]
            if gname_noext.find('.') == -1 and gnameext.find('.') == -1:
                gname = gname_noext + '.' + gnameext
            else: 
                gname = gname_noext + gnameext
            aif   = csv_data[3]
            logdbg('Adding Image (old way)')
        iom.hs.gm.add_img(gid, gname, aif)

    # MOVED INTO CHIP MANAGER READ_CSV_LINE. TODO: do with others
    #def __chip_csv_func(iom, csv_data, csv_headers=None):

    def __name_csv_func(iom, csv_data, csv_headers=None):
        nid  = int(csv_data[0])
        name = (csv_data[1])
        logdbg('Adding Name: '+str(name))
        iom.hs.nm.add_name(nid, name)

    def  load_image_table(iom):
        logmsg('Loading Image Table')
        img_table_fpath = iom.get_image_table_fpath()
        if not exists(img_table_fpath): 
            img_table_fpath = iom._check_altfname(alt_names=['image_table.csv'])
        image_csv_func   = lambda f,d: iom.__image_csv_func(f,d)
        return iom._load_table(img_table_fpath, 'Image', iom.hs.gm.img_alloc, image_csv_func)

    def  load_name_table(iom):
        logmsg('Loading Name Table')
        name_table_fpath = iom.get_name_table_fpath()
        if not exists(name_table_fpath): 
            name_table_fpath = iom._check_altfname(alt_names=['name_table.csv'])
        name_csv_func   = lambda f,d: iom.__name_csv_func(f,d)
        return iom._load_table(name_table_fpath, 'Name', iom.hs.nm.name_alloc, name_csv_func)
    
    def  load_chip_table(iom):
        logmsg('Loading Chip Table')
        chip_table_fpath = iom.get_chip_table_fpath()
        if not exists(chip_table_fpath): 
            alt_names=['chip_table.csv','instance_table.csv','animal_info_table.csv','SightingData.csv']
            chip_table_fpath = iom._check_altfname(alt_names=alt_names)
        return iom._load_table(chip_table_fpath, 'Chip', iom.hs.cm.chip_alloc, iom.hs.cm.load_csv_line)
        #csv_fpath=chip_table_fpath
        #table_name='Chip'
        #alloc_func=iom.hs.cm.chip_alloc
        #csv_func=iom.hs.cm.load_csv_line

    def _check_altfname(iom, alt_names=None):
        'Checks for a legacy data table'
        alt_dirs = [iom.get_internal_dpath(),
                    iom.hs.db_dpath,
                    join(iom.hs.db_dpath,'data'),
                    join(iom.hs.db_dpath,'data','..','data','..')]
        for adir in iter(alt_dirs):
            for aname in iter(alt_names):
                alt_fpath = normpath(join(adir,aname))
                logdbg('Checking: '+alt_fpath)
                if exists(alt_fpath):
                    logwarn('Using Alternative Datatable '+alt_fpath)
                    timestamp = str(time.time())
                    backup_fpath = normpath(alt_fpath+'.'+timestamp+'.bak')
                    logwarn('Creating Backup: '+backup_fpath)
                    shutil.copyfile(alt_fpath, backup_fpath)
                    return alt_fpath
        if iom.hs.db_dpath.find(iom.internal_dname) >= 0:
            # Disallow Hotspotter directories inside HotSpotter directories
            new_db_path = iom.hs.db_dpath[0:iom.hs.db_dpath.find(iom.internal_dname)]
            logwarn('Changing this data dir '+iom.hs.db_dpath)
            logwarn('To that data dir '+new_db_path)
            iom.hs.db_dpath = new_db_path

        return 'CSV_Name_not_found'

    def save_tables(iom):
        hs = iom.hs
        gm = hs.gm
        cm = hs.cm
        nm = hs.nm
        logmsg('Saving the Database. Give it a sec.')
        chip_table_fpath = iom.get_chip_table_fpath()
        name_table_fpath = iom.get_name_table_fpath()
        img_table_fpath = iom.get_image_table_fpath()
        flat_table_fpath = iom.get_flat_table_fpath()

        logmsg('Saving Image Table')
        img_file = open(img_table_fpath, 'w')
        img_file.write(gm.gx2_info(lbls = ['gid','gname','aif']))
        img_file.close()

        logmsg('Saving Name Table')
        name_file = open(name_table_fpath, 'w')
        name_file.write(nm.nx2_info(lbls = ['nid', 'name']))
        name_file.close()

        logmsg('Saving Chip Table')
        chip_file = open(chip_table_fpath, 'w')
        chip_file.write(cm.cx2_info(lbls='all'))
        chip_file.close()

        logmsg('Saving Flat Table')
        flat_file = open(flat_table_fpath, 'w')
        flat_lbls = ['cid','gname','name', 'roi', 'theta'] + cm.user_props.keys()
        flat_file.write(cm.cx2_info(lbls=flat_lbls))
        flat_file.close()
        logmsg('The Database was Saved')


    def get_oxford_gt_dpath(iom):
        return join(iom.hs.db_dpath, 'oxford_style_gt')
    def load_oxford_gt(iom):
        'loads oxford style groundtruth'
        gm,cm,nm = iom.hs.get_managers('gm','cm','nm')
        # Check for corrupted files (Looking at your Paris Buildings Dataset)
        oxford_gt_dpath = iom.get_oxford_gt_dpath()
        corrupted_gname_list = []
        corrupted_file_fname = 'corrupted_files.txt'
        corrupted_file_fpath = join(oxford_gt_dpath,corrupted_file_fname)
        if exists(corrupted_file_fpath):
            with open(corrupted_file_fpath) as f:
                corrupted_gname_list = f.read().splitlines()
        logmsg('Loading Oxford Style Images')
        #Recursively get relative path of all files in img_dpath
        img_dpath  = iom.get_img_dpath() #with a sexy list comprehension
        gname_list = [join(relpath(root, img_dpath), fname).replace('\\','/').replace('./','')\
                      for (root,dlist,flist) in os.walk(img_dpath)\
                      for fname in flist]
        #Roughly Prealloc
        gm.img_alloc( len(gname_list))
        cm.chip_alloc(len(gname_list))
        #Add all images in images directory (allow nested directories (...paris))
        for gname in gname_list:
            if gname in corrupted_gname_list: continue
            gm.add_img(-1, gname, True)

        logmsg('Loading Oxford Style Names and Chips')
        # Add names and chips from ground truth
        gt_fname_list = os.listdir(oxford_gt_dpath)
        iom.hs.nm.name_alloc(len(gt_fname_list)/4)
        for gt_fname in gt_fname_list:
            if gt_fname == corrupted_file_fname: continue
            #Get gt_name, quality, and num from fname
            gt_name = gt_fname.replace('.txt','')
            _pos1 = gt_name.rfind('_')
            quality = gt_name[_pos1+1:]
            gt_name = gt_name[:_pos1]
            _pos2 = gt_name.rfind('_')
            num = gt_name[_pos2+1:]
            gt_name = gt_name[:_pos2]
            # Add Name (-2 suppresses warnings)
            nid = nm.add_name(-2, gt_name)
            nx  = nm.nid2_nx[nid]
            gt_fpath = join(oxford_gt_dpath, gt_fname)
            with open(gt_fpath,'r') as f:
                line_list = f.read().splitlines()
                for line in line_list:
                    if line == '': continue
                    fields = line.split(' ')
                    gname = fields[0].replace('oxc1_','')+'.jpg'
                    if gname.find('paris_') >= 0: 
                        # PARIS HACK >:(
                        #Because they just cant keep their paths consistent 
                        paris_hack = gname[6:gname.rfind('_')]
                        gname = paris_hack+'/'+gname

                    if gname in corrupted_gname_list: continue
                    gid = gm.gname2_gid[gname]
                    gx  = gm.gid2_gx[gid]
                    if len(fields) > 1: #quality == query
                        roi = map(lambda x: int(round(float(x))),fields[1:])
                    else: # quality in ['good','ok','junk']
                        (w,h) = gm.gx2_img_size(gx)
                        roi = [0,0,w,h]
                    cm.add_chip(-1, nx, gx, roi)
        # HACKISH Duplicate detection. Eventually this should actually be in the codebase
        logmsg('Detecting and Removing Duplicate Ground Truth')
        dup_cx_list = []
        for nx in nm.get_valid_nxs():
            cx_list = array(nm.nx2_cx_list[nx])
            gx_list = cm.cx2_gx[cx_list]
            (unique_gx, unique_x) = np.unique(gx_list, return_index=True)
            name = nm.nx2_name[nx]
            for gx in gx_list[unique_x]:
                bit = False
                gname = gm.gx2_gname[gx]
                x_list = pylab.find(gx_list == gx)
                cx_list2  = cx_list[x_list]
                roi_list2 = cm.cx2_roi[cx_list2]
                roi_hash = lambda roi: roi[0]+roi[1]*10000+roi[2]*100000000+roi[3]*1000000000000
                (_, unique_x2) = np.unique(map(roi_hash, roi_list2), return_index=True)
                non_unique_x2 = np.setdiff1d(np.arange(0,len(cx_list2)), unique_x2)
                for nux2 in non_unique_x2:
                    cx_  = cx_list2[nux2]
                    dup_cx_list += [cx_]
                    roi_ = roi_list2[nux2]
                    logmsg('Duplicate: cx=%4d, gx=%4d, nx=%4d roi=%r' % (cx_, gx, nx, roi_) )
                    logmsg('           Name:%s, Image:%s' % (name, gname) )
                    bit = True
                if bit:
                    logmsg('-----------------')
        for cx in dup_cx_list:
            cm.remove_chip(cx)



