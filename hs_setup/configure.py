from fnmatch import fnmatchcase
from hs_setup.git_helpers import *
from os.path import dirname, realpath, join, exists, normpath, isdir, isfile
import os
import shutil
import subprocess
import sys

def normalize_str(instr):
    outstr = instr
    outstr = outstr.replace('\r\n','\n')
    outstr = outstr.replace('\r','\n')
    outstr = outstr.replace('\t','    ')
    return outstr

def clean_git_config():
    'Code for removing old submodules from your config'
    import re
    print 'Cleaning Git Config'
    config_fpath = '../.git/config'
    gitconfig = open(config_fpath,'r').read()
    gitconfig = normalize_str(gitconfig)
    gitconfig = re.sub(' *\[submodule \'tpl\'\] *\n[^\n]*tpl-hotspotter.git *\n',
        '', gitconfig, re.MULTILINE)
    open(config_fpath,'w').write(gitconfig)

def execute_syscalls(syscalls):
    print 'Executing Commands: '
    for _cmd in syscalls.split('\n'):
        cmd = _cmd.strip(' ')
        if cmd == '': continue
        print '  '+cmd
        os.system(cmd)

def setup_submodules():
    server  = 'https://github.com/Erotemic/'
    # User Private Server if available
    if git_fetch_url() == 'git@hyrule.cs.rpi.edu:hotspotter.git':
        server = 'git@hyrule.cs.rpi.edu:'
    execute_syscalls('''
    git submodule add '''+server+'''tpl-hotspotter.git hotspotter/tpl
    git submodule update --init
    git submodule init 
    git submodule update''')

def fix_tpl_permissions():
    execute_syscalls('''
    chmod +x hotspotter/tpl/lib/darwin/*.mac
    chmod +x hotspotter/tpl/lib/linux2/*.ln''')

def compile_widgets():
    if sys.platform == 'win32':
        pyuic4_cmd = r'C:\Python27\Lib\site-packages\PyQt4\pyuic4'
    else:
        pyuic4_cmd = 'pyuic4'
    widget_dir = join(dirname(realpath(__file__)), '../hotspotter/front')
    widget_list = ['MainSkel', 'ChangeNameDialog', 'EditPrefSkel', 'ResultDialog']
    for widget in widget_list:
        widget_ui = join(widget_dir, widget+'.ui')
        widget_py = join(widget_dir, widget+'.py')
        execute_syscalls(pyuic4_cmd+' -x '+widget_ui+' -o '+widget_py)

def configure():
    import os
    os.chdir(dirname(realpath(__file__)))
    #python main.py --delete-preferences
    clean_git_config()
    setup_submodules()
    fix_tpl_permissions()


# Copy boostpython dll to site-packages
def setup_boost():
    print('Setting up Boost')
    boost_root = 'C:/boost_1_53_0'
    boost_lib = boost_root + '/stage/lib'
    python_root = 'C:/Python27'
    site_packages = python_root +'/Lib/site-packages'
    #copy(boost_lib+'/libboost_python-mgw46-mt-1_53.dll',
    #site_packages+'/libboost_python-mgw46-mt-1_53.dll')
    INCLUDE_DIRS = [
        site_packages+'/numpy/core/include',
        boost_root
    ]
    INCLUDE_LIBS = [
        boost_lib
    ]
    INCLUDE_DIRS_STR = ','.join(map(normpath, INCLUDE_DIRS))
    INCLUDE_LIBS_STR = ','.join(map(normpath, INCLUDE_LIBS))

    # Goes in C:\Python27\Lib\distutils\distutils.cfg
    windows_distutils_cfg = '''
    [build]
    compiler=mingw32
    [build_ext]
    include_dirs=%(include_dirs)s
    library_dirs=%(library_dirs)s
    ''' % {
        'include_dirs':INCLUDE_DIRS_STR,
        'library_dirs':INCLUDE_LIBS_STR
    }
    write_text(python_root+'\Lib\distutils\distutils.cfg', windows_distutils_cfg)
