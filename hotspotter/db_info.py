from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile) = __common__.init(__name__, '[dbinfo]')
# Python
import os
import sys
from os.path import isdir, islink, isfile, join, exists
from collections import OrderedDict
import fnmatch
# Science
import numpy as np
from PIL import Image
# Hotspotter
import load_data2 as ld2
from hscom import helpers
from hscom import helpers as util


def dir_size(path):
    if sys.platform == 'win32':
        pass
    else:
        import commands
        size = commands.getoutput('du -sh ' + path).split()[0]
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
        rss = helpers.RedirectStdout()
        rss.start()
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

    def get_db_types(self):
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
        if skip:
            return
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
            print(' * ' + str(dir_))

    def print_databases(self, indent):
        for db_stats in self.db_stats_list:
            print(indent + db_stats.name())
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
    internal_dir = join(path, '.hs_internals')
    tables = [join(internal_dir, 'chip_table.csv'),
              join(internal_dir, 'name_table.csv'),
              join(internal_dir, 'image_table.csv')]
    return all([exists(path_) for path_ in tables])


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
    ss_data = join(path, 'SightingData.csv')
    return helpers.checkpath(ss_data, verbose=False)


def has_v1_gt(path):
    info_table = join(path, 'animal_info_table.csv')
    return helpers.checkpath(info_table, verbose=False)


def has_v2_gt(path):
    tables = [join(path, 'image_table.csv'),
              join(path, 'instance_table.csv')]
    return all([exists(path_) for path_ in tables])


def has_partial_gt(path):
    internal_dir = join(path, '.hs_internals')
    useful_files = ['flat_table.csv',
                    join(internal_dir, '/chip_table.csv'),
                    'chip_table.csv',
                    join(internal_dir, 'instance_table.csv'),
                    'instance_table.csv']
    return any([exists(path_) for path_ in useful_files])


def has_flat_table(path):
    tables = [join(path, '/flat_table.csv')]
    return all([exists(path_) for path_ in tables])


def has_tables(path):
    tables = [
        join(path, '/chip_table.csv'),
        join(path, '/image_table.csv')]
    return all([exists(path_) for path_ in tables])


def has_xlsx_gt(path):
    fpath_list = os.listdir(path)
    xlsx_files = [fpath for fpath in fpath_list if fnmatch.fnmatch(fpath, '*.xlsx')]
    return len(xlsx_files) > 0
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
        if not exists(join(path, 'images')):
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
    nx2_nChips[0:2] = 0  # remove uniden names
    # Seperate singleton / multitons
    multiton_nxs,  = np.where(nx2_nChips > 1)
    singleton_nxs, = np.where(nx2_nChips == 1)
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)
    # Chip Info
    cx2_roi = hs.tables.cx2_roi
    multiton_cx_lists = nx2_cxs[multiton_nxs]
    multiton_cxs = np.hstack(multiton_cx_lists)
    singleton_cxs = nx2_cxs[singleton_nxs]
    multiton_nx2_nchips = map(len, multiton_cx_lists)
    valid_cxs = hs.get_valid_cxs()
    num_chips = len(valid_cxs)
    # Image info
    gx2_gname  = hs.tables.gx2_gname
    cx2_gx = hs.tables.cx2_gx
    num_images = len(gx2_gname)
    img_list = helpers.list_images(hs.dirs.img_dir, fullpath=True)

    def wh_print_stats(wh_list):
        if len(wh_list) == 0:
            return '{empty}'
        stat_dict = OrderedDict(
            [( 'max', wh_list.max(0)),
             ( 'min', wh_list.min(0)),
             ('mean', wh_list.mean(0)),
             ( 'std', wh_list.std(0))])
        arr2str = lambda var: '[' + (', '.join(map(lambda x: '%.1f' % x, var))) + ']'
        ret = (',\n    '.join(['%r:%s' % (key, arr2str(val)) for key, val in stat_dict.items()]))
        return '{\n    ' + ret + '}'

    def get_img_size_list(img_list):
        ret = []
        for img_fpath in img_list:
            try:
                size = Image.open(img_fpath).size
                ret.append(size)
            except Exception as ex:
                print(repr(ex))
                pass
        return ret

    print('reading image sizes')
    if len(cx2_roi) == 0:
        chip_size_list = []
    else:
        chip_size_list = cx2_roi[:, 2:4]
    img_size_list  = np.array(get_img_size_list(img_list))
    img_size_stats  = wh_print_stats(img_size_list)
    chip_size_stats = wh_print_stats(chip_size_list)
    multiton_stats  = helpers.printable_mystats(multiton_nx2_nchips)

    num_names = len(valid_nxs)
    # print
    info_str = '\n'.join([
        (' DB Info: ' + hs.get_db_name()),
        (' * #Img   = %d' % num_images),
        (' * #Chips = %d' % num_chips),
        (' * #Names = %d' % len(valid_nxs)),
        (' * #Unidentified Chips = %d' % len(uniden_cxs)),
        (' * #Singleton Names    = %d' % len(singleton_nxs)),
        (' * #Multiton Names     = %d' % len(multiton_nxs)),
        (' * #Multiton Chips     = %d' % len(multiton_cxs)),
        (' * Chips per Multiton Names = %s' % (multiton_stats,)),
        (' * #Img in dir = %d' % len(img_list)),
        (' * Image Size Stats = %s' % (img_size_stats,)),
        (' * Chip Size Stats = %s' % (chip_size_stats,)), ])
    print(info_str)
    return locals()


def get_keypoint_stats(hs):
    from hscom import latex_formater as pytex
    hs.dbg_cx2_kpts()
    # Keypoint stats
    cx2_kpts = hs.feats.cx2_kpts
    # Check cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[dbinfo] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    acd = kpts[:, 2:5].T
    scales = np.sqrt(acd[0] * acd[2])
    scales = np.array(sorted(scales))
    tex_scale_stats = pytex.latex_mystats(r'kpt scale', scales)
    tex_nKpts       = pytex.latex_scalar(r'\# kpts', len(kpts))
    tex_kpts_stats  = pytex.latex_mystats(r'\# kpts/chip', cx2_nFeats)
    print(tex_nKpts)
    print(tex_kpts_stats)
    print(tex_scale_stats)
    np.set_printoptions(**_printopts)
    print('[dbinfo] ---/LaTeX --- ')
    return (tex_nKpts, tex_kpts_stats, tex_scale_stats)
