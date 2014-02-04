#!/usr/bin/env python
import subprocess
import re
import shutil
import os
from os.path import exists, join, split, splitext, normpath, abspath


def ensuredir(path):
    if not exists(path):
        os.makedirs(path)


def _cmd(*args):
    print(' '.join(args))
    return subprocess.check_output(args)


def extract_dependent_dylibs(dylib_fpath, filter_regex=None):
    'Extracts the dependent libraries of the input dylib'
    out = _cmd('otool', '-L', dylib_fpath)
    out = [line.strip() for line in out.split('\n')]
    if filter_regex is not None:
        out = filter(lambda line: re.search(filter_regex, line), out)
    dylib_list = [line.split(' ')[0] for line in out]
    return dylib_list


def append_suffix(fpath, suffix):
    'appends sufix like /some/filename<suffix>.ext'
    root, fname = split(fpath)
    name, ext = splitext(fname)
    new_fname = name + suffix + ext
    new_fpath = join(root, new_fname)
    return new_fpath


def get_localize_name_cmd(dylib_fpath, fpath_src):
    fname = split(fpath_src)[1]
    loader_dst = join('@loader_path', fname)
    instname_cmd = ['install_name_tool', '-change', fpath_src, loader_dst, dylib_fpath]
    return instname_cmd


def inspect_dylib(dylib_fpath):
    print(_cmd('otool', '-L', dylib_fpath))


def make_distributable_dylib(dylib_fpath, filter_regex='/opt/local/lib/'):
    'removes absolute paths from dylibs on mac using otool'
    print('[otool] making distributable: %r' % dylib_fpath)
    assert exists(dylib_fpath), 'does not exist dylib_fpath=%r' % dylib_fpath
    loader_path = split(dylib_fpath)[0]
    depends_list = extract_dependent_dylibs(dylib_fpath, filter_regex=filter_regex)

    dependency_moved = False

    # Build task list
    copy_list = []
    instname_list = []
    for fpath_src in depends_list:
        # Skip depenencies which are relative paths
        # they have probably already been fixed
        if not exists(fpath_src):
            continue
        fpath_dst = join(loader_path, split(fpath_src)[1])
        # Only copy if the file doesnt already exist
        if not exists(fpath_dst):
            if re.search(filter_regex, fpath_src):
                dependency_moved = True
            copy_list.append((fpath_src, fpath_dst))
        instname_list.append(get_localize_name_cmd(dylib_fpath, fpath_src))
    # Change input name as well
    instname_list.append(get_localize_name_cmd(dylib_fpath, dylib_fpath))

    # Copy the dependencies to the dylib location
    for (fpath_src, fpath_dst) in copy_list:
        shutil.copy(fpath_src, fpath_dst)

    # Change the dependencies in the dylib
    for instname_cmd in instname_list:
        _cmd(*instname_cmd)
    return dependency_moved


def check_depends_dylib(dylib_fpath, filter_regex='/opt/local/lib/'):
    print('[otool] checking dependencies: %r' % dylib_fpath)
    assert exists(dylib_fpath), 'does not exist dylib_fpath=%r' % dylib_fpath
    depends_list = extract_dependent_dylibs(dylib_fpath, filter_regex=filter_regex)
    loader_path = split(dylib_fpath)[0]
    exists_list = []
    missing_list = []
    missing_abs_list = []
    for fpath in depends_list:
        fixed_fpath = normpath(fpath.replace('@loader_path', loader_path))
        absfpath = abspath(fixed_fpath)
        if exists(absfpath):
            exists_list.append(fpath)
        else:
            missing_list.append(fpath)
            missing_abs_list.append(absfpath)

    if len(exists_list) > 0:
        print('Verified Dependencies: ')
        print('\n'.join(exists_list))
        print('----')
    else:
        print('Nothing exists')

    if len(missing_list) > 0:
        print('Missing Dependencies: ')
        print('\n'.join(missing_list))
        print('----')
        print('Missing Dependencies: (absolute path)')
        print('\n'.join(missing_abs_list))
        print('----')
    else:
        print('Nothing missing')


if __name__ == '__main__':
    #from os.path import expanduser
    #dylib_fpath  = expanduser('~/code/hotspotter/hstpl/extern_feat/libhesaff.dylib')
    import sys
    if len(sys.argv) == 3:
        dylib_fpath = sys.argv[2]
        if sys.argv[1] == 'make_distributable':
            make_distributable_dylib(dylib_fpath, filter_regex='/opt/local/lib/')
        elif sys.argv[1] == 'check_depends':
            check_depends_dylib(dylib_fpath, filter_regex='')
        else:
            print('[otool] unknown command')
    else:
        print('[otool] not enough arguments')
        print(sys.argv)
