#!/usr/env python
import subprocess
import re
import shutil
import os
from os.path import exists, expanduser, join, split, splitext


def ensuredir(path):
    if not exists(path):
        os.makedirs(path)


def _cmd(*args):
    print(' '.join(args))
    subprocess.check_output(args)


def extract_dependent_dylibs(dylib_fpath, filter_regex=None):
    'Extracts the dependent libraries of the input dylib'
    out = subprocess.check_output(['otool', '-L', dylib_fpath])
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


def make_distributable_dylib(dylib_fpath):
    'removes absolute paths from dylibs on mac using otool'
    print('[otool] making distributable: %r' % dylib_fpath)
    assert exists(dylib_fpath), 'the input dylib does not exist'
    output_dir = split(dylib_fpath)[0]
    depends_list = extract_dependent_dylibs(dylib_fpath, filter_regex='opencv')

    # Build task list
    copy_list = []
    instname_list = []
    for fpath_src in depends_list:
        fpath_dst = join(output_dir, split(fpath_src)[1])
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

if __name__ == '__main__':
    # input dylib
    dylib_fpath  = expanduser('~/code/hotspotter/hstpl/extern_feat/libhesaff.dylib')
    make_distributable_dylib(dylib_fpath)
