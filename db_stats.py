from __init__ import *
from experiments import get_db_names_info
import params
import load_data2 as ld2
from os.path import isdir, islink, isfile, join
import helpers


def is_current_hotspotter_database(root_dir):
    img_dir      = root_dir + ld2.RDIR_IMG
    internal_dir = root_dir + ld2.RDIR_INTERNAL 

    chip_table   = internal_dir + '/chip_table.csv'
    name_table   = internal_dir + '/name_table.csv'
    image_table  = internal_dir + '/image_table.csv' # TODO: Make optional
    try:
        helpers.assertpath(img_dir)
        helpers.assertpath(internal_dir)
    except AssertionError as ex:
        return False
    try:
        helpers.assertpath(chip_table)
        helpers.assertpath(name_table)
        helpers.assertpath(image_table)
    except AssertionError as ex:
        return False
    return True

def is_legacy_hotspotter_database(root_dir):
    img_dir      = root_dir + ld2.RDIR_IMG
    internal_dir = root_dir + ld2.RDIR_INTERNAL 

    chip_table   = internal_dir + '/chip_table.csv'
    name_table   = internal_dir + '/name_table.csv'
    image_table  = internal_dir + '/image_table.csv' # TODO: Make optional
    try:
        helpers.assertpath(img_dir)
        helpers.assertpath(internal_dir)
    except AssertionError as ex:
        return False
    return False

def is_database(path):
    path_type = 'link' if islink(path) else 'dir'
    is_current_db = is_current_hotspotter_database(path)
    if not is_current_db:
        is_legacy_db = is_legacy_hotspotter_database(path)
        if not is_legacy_db:
            return False
        db_type = 'legacy'
    else:
        db_type = 'current'
    type_str = db_type + ' database ' + path_type + ':\n * ' + path
    print('-----------')
    print(type_str)
    print_name_info(path)

    return True

def print_name_info(db_dir):
    hs = ld2.HotSpotter()
    rss = helpers.RedirectStdout(); rss.start()
    hs.load_tables(db_dir)
    name_info_dict = get_db_names_info(hs)
    rss.stop()
    name_info = name_info_dict['info_str']
    print name_info

def database_crawl(root_dir):
    name_list = os.listdir(root_dir)
    database_list       = []
    directory_link_list = []
    link_list           = []
    directory_list      = []
    file_list           = []
    for name in name_list:
        path = join(root_dir, name)
        if isdir(path) and not islink(path):
            if is_database(path):
                database_list.append(path)
            else:
                directory_list.append(path)
        elif islink(path):
            if is_database(path):
                directory_link_list.append(path)
            else:
                link_list.append(path)
        elif isfile(path) and not islink(path):
            file_list.append(path)
    pass

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    database_crawl(params.WORK_DIR)
    database_crawl(params.WORK_DIR2)




    

