from __future__ import print_function 
from os.path import expanduser, join, relpath, realpath, normpath, exists, dirname
import os, fnmatch, sys

def preferences_dir():
    return normpath(join(expanduser('~'), '.hotspotter'))

def find_hotspotter_root_dir():
    'Find the HotSpotter root dir even in installed packages'
    sys.stdout.write(' ... Finding hotspotter root')
    root_fpath = realpath(dirname(__file__))
    landmark_fname = '__HOTSPOTTER_ROOT__'
    while True:
        landmark_fpath = join(root_fpath, landmark_fname)
        if exists(landmark_fpath):
            break
        next_root = dirname(root_fpath)
        if next_root == root_fpath:
            raise Exception(' !!! Cannot find hotspotter root')
        root_fpath = next_root
    sys.stdout.write(' ... FOUND: '+str(root_fpath)+'\n')
    return root_fpath

def delete_file(fpath):
    print('Deleting: ' + fpath)
    try:
        os.remove(fpath)
        return True
    except OSError as e:
        logwarn('OSError: ' + str(e) + '\n' + \
                'Could not delete: ' + fpath)
        return False

def delete_pattern(dpath, fname_pattern, recursive=True):
    'Removes all files matching fname_pattern in dpath'
    print('Removing files in directory %r %s' % (dpath, ['', ', Recursively'][recursive]))
    print('Removing files with pattern: %r' % fname_pattern)
    num_removed = 0
    num_matched = 0
    for root, dname_list, fname_list in os.walk(dpath):
        for fname in fnmatch.filter(fname_list, fname_pattern):
            num_matched += 1
            num_removed += delete_file(join(root, fname))
        if not recursive:
            break
    print('Removed %d/%d files' % (num_removed, num_matched))
    return True

def delete_preference_dir():
    'Deletes the preference files in the ~/.hotspotter directory'
    print('Deleting ~/.hotspotter/*')
    delete_pattern(preferences_dir(), '*', recursive=False)

