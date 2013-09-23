from __future__ import division
import os
import sys
from os.path import isdir, islink, isfile, join, exists
import params
import load_data2 as ld2
import helpers
import numpy as np


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
        print(root_dir)
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
            print('----')
            if db_stats.version == '(HEAD)':
                db_stats.print_name_info()
            elif 'images' in db_stats.version:
                print(db_stats.db_dir)
                print('num images: %d' % num_images(db_stats.db_dir))
        pass

#--------------------
def has_internal_tables(path):
    internal_dir = path+'/.hs_internals'
    tables = [
        internal_dir+'/chip_table.csv',
        internal_dir+'/name_table.csv',
        internal_dir+'/image_table.csv']
    return all([exists(path) for path in tables])

def num_images(path):
    num_imgs = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if helpers.matches_image(fname):
                num_imgs += 1
    return num_imgs

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
    # some cx information
    multiton_cxs = nx2_cxs[multiton_nxs]
    singleton_cxs = nx2_cxs[singleton_nxs]
    multiton_nx2_nchips = map(len, multiton_cxs)

    num_chips = np.array(nx2_nChips).sum()
    # print
    info_str = '\n'.join([
    (' Name Info: '+hs.db_name()),
    (' * len(uniden_cxs)    = %d' % len(uniden_cxs)),
    (' * len(all_cxs)       = %d' % num_chips),
    (' * len(valid_nxs)     = %d' % len(valid_nxs)),
    (' * len(multiton_nxs)  = %d' % len(multiton_nxs)),
    (' * len(singleton_nxs) = %d' % len(singleton_nxs)),
    (' * multion_nxs #cxs stats: %r' % helpers.printable_mystats(multiton_nx2_nchips)) ])
    print(info_str)
    return locals()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

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
    for db_stats in dir_stats_list:
        print('--')
        print(db_stats.print_db_stats())

    # Print File Stats
    print('\n\n === All Info === ')
    for db_stats in dir_stats_list:
        print('--'+db_stats.name())
        db_stats.print_databases(' * ')

    print('\n\n === NonDB Dirs === ')
    for db_stats in dir_stats_list:
        print('--'+db_stats.name())
        db_stats.print_nondbdirs()

    # Print File Stats
    print('\n\n === Num File Stats === ')
    for db_stats in dir_stats_list:
        print('--')
        print(db_stats.num_files_stats())
