from __future__ import division, print_function
from os.path import dirname, realpath, join, exists, normpath, isdir, isfile
import subprocess
import os
import sys

# josharian: I doubt that the minimal environment stuff here is
# still needed; it is inherited. This was originally
# an hg_version function borrowed from NumPy's setup.py.
# I'm leaving it in for now because I don't have enough other
# environments to test in to be confident that it is safe to remove.
def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'PYTHONPATH']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        env=env
    ).communicate()[0]
    return out

def git_fetch_url():
    ''' Return the git fetch url.'''
    fetch_url = 'unknown'
    try:
        out = _minimal_ext_cmd(['git', 'remote', '-v']).strip().decode('ascii')
        for item in out.split('\n'):
            fetch_pos = item.find(' (fetch)')
            origin_pos = item.find('origin\t')
            if fetch_pos > -1 and origin_pos > -1:
               fetch_url  = item[origin_pos+7:fetch_pos]
    except Exception:
        fetch_url = 'unknown-exception'
    return fetch_url

def git_branch():
    ''' Return the current git branch. '''
    try:
        out = _minimal_ext_cmd(['git', 'branch'])
        _branch1 = out.strip().decode('ascii')+'\n'
        _branch2 = _branch1[_branch1.find('*')+1:]
        branch   = _branch2[:_branch2.find('\n')].strip()
    except OSError:
        branch = 'release'
    return branch

def git_version():
    ''' Return the sha1 of local git HEAD as a string. '''
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = 'unknown-git'
    return git_revision

