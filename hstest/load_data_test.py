#!/usr/bin/env python
from __future__ import division, print_function
#
from os.path import join
import multiprocessing
import os
# HotSpotter
from hotspotter import load_data3 as ld3
from hotspotter import db_info
# Test load csv tables


def execute_test(db_dir):
    print('--------------------------------')
    print('[TEST] db_dir = %r' % db_dir)
    version_info = ld3.detect_version(db_dir)
    print('[TEST] db_version = %r' % version_info['db_version'])
    (chip_table, name_table, image_table) = version_info['tables_fnames']
    if name_table is not None:
        print('[TEST] name_table = %r' % name_table)
    print('--------------------------------')
    pass

if __name__ == '__main__':
    multiprocessing.freeze_support()
    ld3.rrr()
    db_info.rrr()

    data_dir = '/media/Store/data'
    testdata_dir = join(data_dir, 'tests')

    test_dir_list = os.listdir(testdata_dir)

    for test_dname in test_dir_list:
        db_dir = join(testdata_dir, test_dname)
        execute_test(db_dir)

"""
exec(open('_tests/test_load_data.py').read())
"""
