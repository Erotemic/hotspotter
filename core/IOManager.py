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
import os.path
import shutil
import time
import fnmatch
import tpl
from pylab import find
from os.path import expanduser, join, normpath, relpath
from other.helpers import *
from other.logger           import *
from other.crossplat import safepath, platexec

def checkdir_decorator(method_fn):
    def wrapper(iom, *args):
        ret = method_fn(iom, *args)
        dircheck(ret)
        return ret
    return wrapper

class IOManager(AbstractManager):

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

    def remove_computed_files_with_pattern(iom, fname_pattern):
        'removes files in computed_dpath'
        logdbg('Removing computed files with pattern: %r' % fname_pattern)
        num_removed = 0
        num_matched = 0
        for root, dname_list, fname_list in os.walk(iom.get_computed_dpath()):
            for fname in fnmatch.filter(fname_list, fname_pattern):
                num_matched += 1
                num_removed += iom.remove_file(os.path.join(root, fname))
        logmsg('Removed %d/%d files' % (num_removed, num_matched))
    
    def __init__(iom, hs):
        super( IOManager, iom ).__init__( hs )        
        logdbg('Creating IOManager')
        iom.hs = hs
        iom.global_dir = safepath(join(expanduser('~'),'.hotspotter'))
        iom.internal_dname = '.hs_internals';
        iom.dummy_delete = False #Dont actually delete things

    # NEW AND UNTESTED
    def get_tpl_lib_dir(iom):
        return os.path.join(os.path.dirname(tpl.__file__), 'lib', sys.platform)
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
    def  get_thumb_dpath(iom):
        return join(iom.get_computed_dpath(), 'thumbs')
    def  get_experiment_dpath(iom):
        return join(iom.get_computed_dpath(), 'experiments')
    # --- Public Directories
    @checkdir_decorator
    def  get_img_dpath(iom, thumb_bit=None):
        img_dname = 'images'
        if (thumb_bit != None and thumb_bit) or iom.hs.prefs['thumbnail_bit']:
            return join(iom.get_thumb_dpath(),'images')
        else:
            return join(iom.hs.db_dpath,'images')
    @checkdir_decorator
    def  get_chip_dpath(iom, thumb_bit=None):
        chip_dname = 'chips'
        if (thumb_bit != None and thumb_bit) or iom.hs.prefs['thumbnail_bit']:
            ret = join(iom.get_thumb_dpath(),chip_dname)
        else:
            ret = join(iom.get_computed_dpath(),chip_dname)
        dircheck(ret)
        return ret
    @checkdir_decorator
    def  get_chiprep_dpath(iom):
        return join(iom.get_computed_dpath(), 'features')
    @checkdir_decorator
    def  get_model_dpath(iom):
        return join(iom.get_computed_dpath(),'models')
    @checkdir_decorator
    def  get_temp_dpath(iom):
        return join(iom.get_computed_dpath(), 'temp')
    def  get_temp_fpath(iom, tmp_fname):
        return safepath(join(iom.get_temp_dpath(), tmp_fname))
    # --- Main Saved Files
    def  get_image_table_fpath(iom):
        return safepath(join(iom.get_internal_dpath(),'image_table.csv'))
    def  get_chip_table_fpath(iom):
        return safepath(join(iom.get_internal_dpath(),'chip_table.csv'))
    def  get_name_table_fpath(iom):
        return safepath(join(iom.get_internal_dpath(),'name_table.csv'))
    def  get_flat_table_fpath(iom):
        return safepath(join(iom.hs.db_dpath,'flat_table.csv'))
    # --- Executable Filenames
    def  get_hesaff_exec(iom):
        return platexec(join(iom.get_tpl_lib_dir(), 'hesaff'))
    def  get_inria_exec(iom):
        return platexec(join(iom.get_tpl_lib_dir(), 'inria_features'))
    # --- Chip Representations
    def get_chip_prefix(iom, cid):
        return 'CID.'+str(cid)+'_'
    def  get_chiprep_fpath(iom, cid):
        am = iom.hs.am
        algo_suffix = am.get_algo_suffix(depends=['preproc', 'chiprep'])
        chiprep_fname = iom.get_chip_prefix(cid)+algo_suffix+'_feats.npz' 
        return safepath(join(iom.get_chiprep_dpath(), chiprep_fname))
    def  get_img_fpath(iom, gname, thumb_bit=None):
        return safepath(join(iom.get_img_dpath(thumb_bit), gname))
    def  get_chip_fpath(iom, cid, thumb_bit=None):
        am = iom.hs.am
        imgext = ['png','jpg'][thumb_bit]
        algo_suffix = am.get_algo_suffix(depends=['preproc'])
        chip_fname = iom.get_chip_prefix(cid)+algo_suffix+'_chip.'+imgext 
        return safepath(join(iom.get_chip_dpath(thumb_bit),chip_fname))
    def get_model_fpath(iom):
        am, vm = iom.hs.get_managers('am','vm')
        algo_suffix = am.get_algo_suffix(depends=['preproc','chiprep','model'])
        samp_suffix = vm.get_samp_suffix()
        model_fname = 'model'+samp_suffix+algo_suffix+'.npz'
        return safepath(join(iom.get_model_dpath(),model_fname))
    def get_flann_index_fpath(iom):
        am, vm = iom.hs.get_managers('am','vm')
        algo_suffix = am.get_algo_suffix(['preproc','chiprep','model'])
        samp_suffix = vm.get_samp_suffix()
        flann_index_fname = 'index.%s.%s.flann' % (algo_suffix, samp_suffix)
        return safepath(join(iom.get_model_dpath(), flann_index_fname))
    # --- Indexes 
    def get_prefs_fpath(iom):
        dircheck(iom.global_dir)
        pref_fname = 'prefs.txt'
        return safepath(join(iom.global_dir, pref_fname))

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
        logmsg('Loading '+iom.hs.db_dpath)
        if not (filecheck(iom.get_image_table_fpath()) and\
                filecheck(iom.get_name_table_fpath()) and\
                filecheck(iom.get_image_table_fpath())):
            if os.path.exists(iom.get_oxford_gt_dpath()):
                logmsg('You have selected an Oxford style groundtruth')
                iom.load_oxford_gt()
                logmsg('Succesfully Loaded Oxford style groundtruth')
                sys.stdout.flush()
                return
            logwarn('Trying to load a Legacy Database')
        iom.load_image_table()
        iom.load_name_table()
        iom.load_chip_table()
        logmsg('Done Loading Tables')
        sys.stdout.flush()

    #START: CSV IO METHODS

    #=======================================
    # IO Internals

    # Reads HotSpotter specific csv headers.
    # num) attribute - description
    def  __load_table(iom, csv_fpath, table_name, alloc_func, csv_func):
        logio('Loading '+table_name+' Table: '+csv_fpath)
        if not filecheck(csv_fpath):
            logio('\"'+csv_fpath+'\" Does Not Exist')
            return False
        fid = file(csv_fpath, 'r')
        in_header_bit = True
        data_headers = None
        line = fid.readline()
        while line != '':
            line = line[:-1]
            if line == '\n':
                continue
            elif line[0] == '#':
                if in_header_bit:
                    num_line_prefix = '#NumLines'
                    if line.find(num_line_prefix) > -1:
                        num_line_str = line.replace(num_line_prefix,'').replace(' ','');
                        num_lines = int(num_line_str[0:-1])
                        alloc_func(num_lines)
                    if line.find('#imgindex') > -1:
                        logmsg('Loading a Legacy StripeSpotter File')
                        csv_fields = line[1:].split(',')
                        data_headers = csv_fields
                    if line.find('#01)') > -1:
                        logmsg('Loading a Legacy HotSpotter File')
                        data_headers = []
                        while line != '':
                            line = line[:-1]
                            if len(line) < 4 or line[3] != ')': break
                            parnstr = '#\\d\\d\\) '
                            data_field = re.sub(parnstr, '', line)
                            data_field = re.sub(' - .*','', data_field)
                            data_headers += [data_field]
                            line = fid.readline()
            else:
                csv_fields = line.split(',');
                csv_func(csv_fields, data_headers)
            line = fid.readline()

        fid.close()
        logio('Loaded '+table_name+' Table')
        return True

    def __image_csv_func(iom, csv_fields, data_headers=None):
        csv_fields = map(lambda k: k.strip(' '), csv_fields)
        #gid   = None
        #gname = None
        #aif   = None
        #if data_headers != None: pass
        if len(csv_fields) == 3:
            gid   = int(csv_fields[0])
            gname = csv_fields[1]
            aif   = csv_fields[2]
            logdbg('Adding Image')
        elif len(csv_fields) == 4:
            gid   = int(csv_fields[0])
            gnameext   = csv_fields[2]
            gname_noext = csv_fields[1]
            if gname_noext.find('.') == -1 and gnameext.find('.') == -1:
                gname = gname_noext + '.' + gnameext
            else: 
                gname = gname_noext + gnameext
            aif   = csv_fields[3]
            logdbg('Adding Image (old way)')
        iom.hs.gm.add_img(gid, gname, aif)

    def __chip_csv_func(iom, csv_fields, data_headers=None):
        csv_fields = map(lambda k: k.strip(' '), csv_fields)
        #gid   = None
        #gname = None
        #aif   = None
        if data_headers is None:
            data_headers = ['cid','gid','nid','roi']
        if data_headers != None:
            if len(data_headers) != len(csv_fields):
                logerr('Error reading chip_file. len(data_headers) != len(csv_fields) length mismatch\n'+\
                      str(data_headers)+'\n'+str(data_headers))
            dmap = {}
            for (a,b) in zip(data_headers,csv_fields):
                dmap[a] = b
            if 'imgindex' in dmap.keys():
                logwarn('Found imgindex')
                imgindex = int(dmap['imgindex'])
                gname = 'img-%07d.jpg' % imgindex
                iom.hs.gm.add_img(int(imgindex), gname, False)
                dmap['gid'] = imgindex
                dmap['cid'] = imgindex
                del dmap['imgindex']
            if 'animal_name' in dmap.keys():
                logwarn('Found animal_name')
                dmap['nid'] = iom.hs.nm.add_name(-1, dmap['animal_name'])
                del dmap['animal_name']
            if 'instance_id' in dmap.keys():
                dmap['cid'] = dmap['instance_id']
                del dmap['instance_id']
            if 'image_id' in dmap.keys():
                dmap['gid'] = dmap['image_id']
                del dmap['image_id']
            if 'name_id' in dmap.keys():
                dmap['nid'] = dmap['name_id']
                del dmap['name_id']


        cid = int(dmap['cid'])
        gid = int(dmap['gid'])
        nid = int(dmap['nid'])
        roi_field = re.sub('  *',' ', dmap['roi'].replace(']','').replace('[','')).strip(' ').rstrip()

        roi = map(lambda x: int(round(float(x))),roi_field.split(' '))
        nx  = iom.hs.nm.nid2_nx[nid]
        gx  = iom.hs.gm.gid2_gx[gid]
        logdbg('Adding Chip: (cid=%d),(nid=%d,nx=%d),(gid=%d,gx=%d)' % (cid, nid, nx, gid, gx))
        if gx == 0 or nx == 0 or gid == 0 or nid == 0:
            logmsg('Adding Chip: (cid=%d),(nid=%d,nx=%d),(gid=%d,gx=%d)' % (cid, nid, nx, gid, gx))
            logerr('Chip has invalid indexes')
        iom.hs.cm.add_chip(cid, nx, gx, roi, delete_prev=False)

    def __name_csv_func(iom, csv_fields, data_headers=None):
        csv_fields = map(lambda k: k.strip(' '), csv_fields)
        nid  = int(csv_fields[0])
        name = (csv_fields[1])
        logdbg('Adding Name: '+str(name))
        iom.hs.nm.add_name(nid, name)

    def  load_image_table(iom):
        logmsg('Loading Image Table')
        img_table_fpath = iom.get_image_table_fpath()
        if not filecheck(img_table_fpath): 
            img_table_fpath = iom._check_altfname(alt_names=['image_table.csv'])
        image_csv_func   = lambda f,d: iom.__image_csv_func(f,d)
        image_alloc_func = lambda num: iom.gm.img_alloc(num)
        return iom.__load_table(img_table_fpath, 'Image', image_alloc_func, image_csv_func)

    def  load_name_table(iom):
        logmsg('Loading Name Table')
        name_table_fpath = iom.get_name_table_fpath()
        if not filecheck(name_table_fpath): 
            name_table_fpath = iom._check_altfname(alt_names=['name_table.csv'])
        name_csv_func   = lambda f,d: iom.__name_csv_func(f,d)
        name_alloc_func = lambda num: iom.gm.name_alloc(num)
        return iom.__load_table(name_table_fpath, 'Name', name_alloc_func, name_csv_func)
    
    def  load_chip_table(iom):
        logmsg('Loading Chip Table')
        chip_table_fpath = iom.get_chip_table_fpath()
        if not filecheck(chip_table_fpath): 
            alt_names=['chip_table.csv','instance_table.csv','animal_info_table.csv','SightingData.csv']
            chip_table_fpath = iom._check_altfname(alt_names=alt_names)
        chip_csv_func   = lambda f,d: iom.__chip_csv_func(f,d)
        chip_alloc_func = lambda num: iom.cm.chip_alloc(num)
        return iom.__load_table(chip_table_fpath, 'Chip', chip_alloc_func, chip_csv_func)

    def _check_altfname(iom, alt_names=None):
        'Checks for a legacy data table'
        alt_dirs = [iom.get_internal_dpath(),
                    iom.hs.db_dpath,
                    join(iom.hs.db_dpath,'data'),
                    join(iom.hs.db_dpath,'data','..','data','..')]
        for adir in iter(alt_dirs):
            for aname in iter(alt_names):
                alt_fpath = safepath(join(adir,aname))
                logmsg('Checking: '+alt_fpath)
                if filecheck(alt_fpath):
                    logwarn('Using Alternative Datatable '+alt_fpath)
                    timestamp = str(time.time())
                    backup_fpath = safepath(alt_fpath+'.'+timestamp+'.bak')
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
        chip_file.write(cm.cx2_info(lbls=['cid','gid','nid','roi']))
        chip_file.close()

        logmsg('Saving Flat Table')
        flat_file = open(flat_table_fpath, 'w')
        flat_file.write(cm.cx2_info(lbls=['cid','gname','name', 'roi']))
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
        if os.path.exists(corrupted_file_fpath):
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
        import numpy as np
        dup_cx_list = []
        for nx in nm.get_valid_nxs():
            cx_list = array(nm.nx2_cx_list[nx])
            gx_list = cm.cx2_gx[cx_list]
            (unique_gx, unique_x) = np.unique(gx_list, return_index=True)
            name = nm.nx2_name[nx]
            for gx in gx_list[unique_x]:
                bit = False
                gname = gm.gx2_gname[gx]
                x_list = find(gx_list == gx)
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


