from __future__ import division, print_function
import __builtin__
import sys
# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass
# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[dbinfo] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()
import os
import sys
from os.path import isdir, islink, isfile, join, exists
import params
import load_data2 as ld2
import helpers
import numpy as np
from PIL import Image

def dir_size(path):
    if sys.platform == 'win32':
        pass
    else:
        import commands
        size = commands.getoutput('du -sh '+path).split()[0]
    return size

def is_dir2(path):
    return isdir(path) and not islink(path)
def is_link2(path):
    return islink(path) and isdir(path)

class DatabaseStats(object):
    def __init__(self, db_dir, version, root_dir):
        self.version = version
        self.db_dir = db_dir
        self.root_dir = root_dir

    def name(self):
        simlink_suffix = '[symlink]' if islink(self.db_dir) else ''
        db_name  = os.path.relpath(self.db_dir, self.root_dir)
        #os.path.split(self.db_dir)[1]
        name_str = '%s %s %s' % (db_name, self.version, simlink_suffix)
        return name_str

    def print_name_info(self):
        hs = ld2.HotSpotter()
        rss = helpers.RedirectStdout(); rss.start()
        hs.load_tables(self.db_dir)
        name_info_dict = get_db_names_info(hs)
        rss.stop()
        name_info = name_info_dict['info_str']
        print(name_info)

KNOWN_BASE_DIRS = [
    'Camera-Traps'
]

KNOWN_DATA_DIRS = [
    '101_ObjectCategories',
    'inria-horses-v103',
    'zebra photo\'s with mothers',
    'zebras-with-mothers-rose-nathan',
    'zebra_with_mothers',
    'Oxford100k',
]

class DirectoryStats(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.db_stats_list = []
        self.dir_list  = []
        self.file_list = []
        self.link_list = []
        #
        self.non_leaves = []
        self.build_self(root_dir)
        #---

    def get_db_types():
        if 'db_types' in self.__dict__.keys():
            return self.db_types
        self.db_types = []


    def build_self(self, root_dir):
        print('root_dir=%r' % root_dir)
        # Do not build a data or db_dir
        if os.path.split(root_dir)[1] in KNOWN_DATA_DIRS:
            return
        name_list = os.listdir(root_dir)
        for name in name_list:
            path = join(root_dir, name)
            self.add_path(path)
        non_leaves = self.non_leaves[:]
        self.non_leaves = []
        # Do not recurse into a base directory
        skip = any([skip_dir in root_dir for skip_dir in KNOWN_BASE_DIRS])
        if skip: return
        for branch in non_leaves:
            #continue
            self.build_self(branch)

    def add_path(self, path):
        db_version = get_database_version(path)
        if not db_version is None: 
            db_stats = DatabaseStats(path, db_version, self.root_dir)
            self.db_stats_list.append(db_stats)
        elif is_not_leaf(path):
            self.non_leaves += [path]
        elif is_dir2(path):
            self.dir_list += [path]
        elif is_link2(path):
            self.link_list += [path]
        elif isfile(path):
            self.file_list += [path]
        else:
            print('path=%r' % path)
            print('path is not addable')
            assert False

    def num_files_stats(self):
        toret = '\n'.join([
            ('Num File Stats: %r' % self.root_dir),
            ('# DB: %r; ' % len(self.db_stats_list)),
            ('# Non-DB files: %r' % len(self.file_list)),
            ('# Non-DB dirs:  %r' % len(self.dir_list)),
            ('# Non-DB links: %r' % len(self.link_list))
        ])
        return toret

    def name(self):
        return self.root_dir

    def print_nondbdirs(self):
        for dir_ in self.dir_list:
            print(' * '+str(dir_))

    def print_databases(self, indent):
        for db_stats in self.db_stats_list:
            print(indent+db_stats.name())
        pass

    def print_db_stats(self):
        for db_stats in self.db_stats_list:
            print_database_stats(db_stats)
        pass

def print_database_stats(db_stats):
    'Prints a single dbstats object'
    print('----')
    if db_stats.version == '(HEAD)':
        db_stats.print_name_info()
    elif 'images' in db_stats.version:
        print(db_stats.db_dir)
        print('num images: %d' % helpers.num_images_in_dir(db_stats.db_dir))

#--------------------
def has_internal_tables(path):
    internal_dir = path+'/.hs_internals'
    tables = [
        internal_dir+'/chip_table.csv',
        internal_dir+'/name_table.csv',
        internal_dir+'/image_table.csv']
    return all([exists(path) for path in tables])

def is_imgdir(path):
    if not isdir(path):
        return False
    img_dir = path + '/images'
    if exists(img_dir):
        return True
    files = os.listdir(path)
    num_files = 0
    num_imgs = 0
    num_dirs = 0
    for name in files:
        subpath = join(path, name)
        if helpers.matches_image(subpath):
            num_imgs += 1
            return True
        elif isdir(subpath):
            num_dirs += 1
        elif isfile(subpath):
            num_files += 1
    return False

def has_ss_gt(path):
    ss_data = path+'/SightingData.txt'
    return exists(ss_data)

def has_v1_gt(path):
    info_table = path+'/animal_info_table.csv'

def has_v2_gt(path):
    tables = [
        path+'/image_table.csv',
        path+'/instance_table.csv']
    return all([exists(path) for path in tables])

def has_partial_gt(path):
    internal_dir = path+'/.hs_internals'
    tables = [
        'flat_table.csv', 
        internal_dir+'/chip_table.csv', 
        'chip_table.csv',
        internal_dir+'instance_table.csv',
        'instance_table.csv']
    return any([exists(path) for path in tables])

def has_flat_table(path):
    tables = [path + '/flat_table.csv']
    return all([exists(path) for path in tables])

def has_tables(path):
    tables = [
        path+'/chip_table.csv',
        path+'/image_table.csv']
    return all([exists(path) for path in tables])
#--------------------

def get_database_version(path):
    is_head = has_internal_tables(path)
    is_ss = has_ss_gt(path)
    is_v1 = has_v1_gt(path)
    is_v2 = has_v2_gt(path)
    if is_head:
        version = 'HEAD'
    elif is_ss or is_v1 or is_v2:
        version = 'legacy'
    elif has_partial_gt(path):
        version = 'partial'
    elif is_imgdir(path):
        version = 'images'
    elif os.path.split(path)[1] in KNOWN_DATA_DIRS:
        version = 'known'
    else:
        version = None

    if not version is None:
        if not exists(path+'/images'):
            version += '+broken'
        version = '(%s)' % version
    return version

def is_not_leaf(path):
    if not isdir(path) or islink(path):
        return False
    names = os.listdir(path)
    for name in names:
        path = join(path, name)
        if isdir(path):
            return True
    return False

def get_db_names_info(hs):
    return db_info(hs)

def db_info(hs):
    # Name Info
    nx2_cxs    = np.array(hs.get_nx2_cxs())
    nx2_nChips = np.array(map(len, nx2_cxs))
    uniden_cxs = np.hstack(nx2_cxs[[0, 1]])
    num_uniden = nx2_nChips[0] + nx2_nChips[1] 
    nx2_nChips[0:2] = 0 # remove uniden names
    # Seperate singleton / multitons
    multiton_nxs,  = np.where(nx2_nChips > 1)
    singleton_nxs, = np.where(nx2_nChips == 1)
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs]) 
    num_names_with_gt = len(multiton_nxs)
    # Chip Info
    cx2_roi = hs.tables.cx2_roi
    multiton_cxs = nx2_cxs[multiton_nxs]
    singleton_cxs = nx2_cxs[singleton_nxs]
    multiton_nx2_nchips = map(len, multiton_cxs)
    valid_cxs = hs.get_valid_cxs()
    num_chips = len(valid_cxs)
    # Image info
    gx2_gname  = hs.tables.gx2_gname
    cx2_gx     = hs.tables.cx2_gx
    num_images = len(gx2_gname)
    img_list = helpers.list_images(hs.dirs.img_dir, fullpath=True)
    def wh_print_stats(wh_list):
        from collections import OrderedDict
        if len(wh_list) == 0:
            return '{empty}'
        stat_dict = OrderedDict(
            [( 'max', wh_list.max(0)),
             ( 'min', wh_list.min(0)),
             ('mean', wh_list.mean(0)),
             ( 'std', wh_list.std(0))])
        arr2str = lambda var: '['+(', '.join(map(lambda x: '%.1f' % x, var)))+']'
        ret = (',\n    '.join(['%r:%s' % (key, arr2str(val)) for key, val in stat_dict.items()]))
        return '{\n    ' + ret +'}'
    def get_img_size_list(img_list):
        ret = []
        for img_fpath in img_list:
            try:
                size = Image.open(img_fpath).size
                ret.append(size)
            except Exception as ex:
                pass
        return ret

    print('reading image sizes')
    if len(cx2_roi) == 0:
        chip_size_list = []
    else:
        chip_size_list = cx2_roi[:,2:4]
    img_size_list  = np.array(get_img_size_list(img_list))
    img_size_stats  = wh_print_stats(img_size_list)
    chip_size_stats = wh_print_stats(chip_size_list)
    multiton_stats  = helpers.printable_mystats(multiton_nx2_nchips)

    # print
    info_str = '\n'.join([
    (' DB Info: '+hs.db_name()),
    (' * #Img   = %d' % num_images),
    (' * #Chips = %d' % num_chips),
    (' * #Names = %d' % len(valid_nxs)),
    (' * #Unidentified Chips = %d' % len(uniden_cxs)),
    (' * #Singleton Names    = %d' % len(singleton_nxs)),
    (' * #Multiton Names     = %d' % len(multiton_nxs)),
    (' * #Multiton Chips     = %d' % len(np.hstack(multiton_cxs))),
    (' * Chips per Multiton Names = %s' % (multiton_stats,)), 
    (' * #Img in dir = %d' % len(img_list)),
    (' * Image Size Stats = %s' % (img_size_stats,)),
    (' * Chip Size Stats = %s' % (chip_size_stats,)),])
    print(info_str)
    return locals()

if __name__ == '__main__':
    #import multiprocessing
    #np.set_printoptions(threshold=5000, linewidth=5000)
    #multiprocessing.freeze_support()
    #print('[dev]-----------')
    #print('[dev] main()')
    #df2.DARKEN = .5
    #main_locals = iv.main()
    #exec(helpers.execstr_dict(main_locals, 'main_locals'))
    from multiprocessing import freeze_support
    freeze_support()

    if sys.argv > 1: 
        import params
        import sys
        path = params.DEFAULT
        db_version = get_database_version(path)
        print('db_version=%r' % db_version)
        if not db_version is None: 
            db_stats = DatabaseStats(path, db_version, params.WORK_DIR)
            print_database_stats(db_stats)
        sys.exit(0)


    # Build list of directories with database in them
    root_dir_list = [
        params.WORK_DIR,
        params.WORK_DIR2
    ]
    DO_EXTRA = True #False
    if sys.platform == 'linux2' and DO_EXTRA:
        root_dir_list += [
            #'/media/Store/data/raw',
            #'/media/Store/data/gold',
            '/media/Store/data/downloads']

    # Build directory statistics
    dir_stats_list = [DirectoryStats(root_dir) for root_dir in root_dir_list]

    # Print Name Stats
    print('\n\n === Num File Stats === ')
    for dir_stats in dir_stats_list:
        print('--')
        print(dir_stats.print_db_stats())

    # Print File Stats
    print('\n\n === All Info === ')
    for dir_stats in dir_stats_list:
        print('--'+dir_stats.name())
        dir_stats.print_databases(' * ')

    print('\n\n === NonDB Dirs === ')
    for dir_stats in dir_stats_list:
        print('--'+dir_stats.name())
        dir_stats.print_nondbdirs()

    # Print File Stats
    print('\n\n === Num File Stats === ')
    for dir_stats in dir_stats_list:
        print('--')
        print(dir_stats.num_files_stats())


