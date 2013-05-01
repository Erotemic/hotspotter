#!/usr/bin/env python
# http://docs.python.org/2/distutils/setupscript.html
# 
# This isn't working yet, I just took the package from Theno and
# changed some variables. I haven't understood how to work this yet.
import os
import sys
import subprocess
from fnmatch import fnmatchcase
from distutils.util import convert_path
#try:
from setuptools import setup
#except ImportError:
    #from distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py
    from distutils.command.build_scripts import build_scripts
#else:
    #exclude_fixers = ['fix_next', 'fix_filter']
    #from distutils.util import Mixin2to3
    #from lib2to3.refactor import get_fixers_from_package
    #Mixin2to3.fixer_names = [f for f in get_fixers_from_package('lib2to3.fixes')
                             #if f.rsplit('.', 1)[-1] not in exclude_fixers]
    #from distutils.command.build_scripts import build_scripts_2to3 as build_scripts

INSTALL_REQUIRES = \
[
    'numpy>=1.5.0',
    'scipy>=0.7.2',
    'PIL>=1.1.7'
]

INSTALL_OPTIONAL = \
[
    'python-qt>=.50',
    'matplotlib>=1.2.1rc1',
    'pyvlfeat>=0.1.1a3'
]

INSTALL_OTHER = \
[
    'boost-python>=1.52',
    'Cython>=.18',
    'ipython>=.13.1'
]

INSTALL_BUILD = \
[
    'Cython'
]

INSTALL_DEV = \
[
    'py2exe>=0.6.10dev',
    'pyflakes>=0.6.1',
    'pylint>=0.27.0',
    'RunSnakeRun>=2.0.2b1',
    'maliae>=0.4.0.final.0',
    'pycallgraph>=0.5.1'
    'coverage>=3.6'
]

CLASSIFIERS = """\
Development Status :: 1 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""
NAME                = 'HotSpotter'
AUTHOR              = "Jonathan Crall, RPI"
AUTHOR_EMAIL        = "crallj@rpi.edu"
MAINTAINER          = AUTHOR
MAINTAINER_EMAIL    = AUTHOR_EMAIL
DESCRIPTION         = 'Image Search for Large Animal Databases.'
LONG_DESCRIPTION    = open("DESCRIPTION.txt").read()
URL                 = "http://www.cs.rpi.edu/~cralljp"
DOWNLOAD_URL        = "https://github.com/Erotemic/hotspotter/archive/release.zip"
LICENSE             = 'BSD'
PLATFORMS           = ["Windows", "Linux", "Mac OS-X"]
MAJOR               = 0
MINOR               = 0
MICRO               = 0
SUFFIX              = "rc3"  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False
VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)

def setup_windows():
    setup_submodules()

def setup_submodules():
    private = 'git@hyrule.cs.rpi.edu:'
    public = 'https://github.com/Erotemic/'
    if not os.path.exists('hotspotter/tpl'):
        os.system('git submodule add '+private+'tpl-hotspotter.git hotspotter/tpl')

    os.system('git submodule update --init')


# Copy boostpython dll to site-packages
def setup_boost():
    print('Setting up Boost')
    boost_root = 'C:/boost_1_53_0'
    boost_lib = boost_root + '/stage/lib'
    python_root = 'C:/Python27'
    site_packages = python_root +'/Lib/site-packages'
    #copy(boost_lib+'/libboost_python-mgw46-mt-1_53.dll', site_packages+'/libboost_python-mgw46-mt-1_53.dll')
    INCLUDE_DIRS = [
        site_packages+'/numpy/core/include',
        boost_root
    ]
    INCLUDE_LIBS = [
        boost_lib
    ]
    INCLUDE_DIRS_STR = ','.join(map(os.path.normpath, INCLUDE_DIRS))
    INCLUDE_LIBS_STR = ','.join(map(os.path.normpath, INCLUDE_LIBS))

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

# Installation Steps that I took: 
#http://ctrl-dev.com/2012/02/compiling-boost-python-with-mingw/
# Installed MinGW
# 
# Installed Boost.Python
# wget(http://sourceforge.net/settings/mirror_choices?projectname=boost&filename=boost/1.53.0/boost_1_53_0.zip)
# unzip boost_1_53_0.zip C:\boost_1_53_0
# cd C:\boost_1_53_0
# bootstrap.bat mingw
# .\b2 
#
# Then Create a File in your boost directory
# user-config.jam 
'''
import toolset : using ;
using python : 2.7 : "C:/Python27" : "C:/Python27/include" : "C:/Python27/libs" ;
'''
#bjam toolset=gcc link=shared --with-python --user-config=user-config.jam
#
# pyvlfeat has the wrong name for boost_python. Fix it

# Ok, try doing boost with
# bjam toolset=msvc-10.0 link=shared --build-type=complete --with-thread
# bjam toolset=gcc link=shared --build-type=complete --with-thread --with-python --with-system

# 
#import subprocess
#def make_link(target='', dest=''):
    #try: 
        #if os.path.exists(dest):
            #print 'Already Exists Error: Destination File: '+dest
            #return
        #if not os.path.exists(target):
            #print 'Doesnt Exists Error: Target File: '+dest
            #return
        #if os.path.isfile(target):
            #command = 'mklink /H '+dest+' '+target
            #print subprocess.call(command)
        #if os.path.isdir(target):
            #command = 'mklink /D '+dest+' '+target
            #print subprocess.call(command)
    #except Exception as ex:
        #print 'Make Link Failed: Error:\n'+str(ex)
        #raise ex
    #make_link(\
        #target=r'C:\boost_1_53_0\stage\lib\libboost_python-mgw46-mt-1_53.dll', 
        #dest=r'C:\boost_1_53_0\stage\lib\boost_python-mt-py26.dll')
    #make_link(\
        #target=r'C:\boost_1_53_0\stage\lib\libboost_python-mgw46-mt-1_53.dll.a', 
        #dest=r'C:\boost_1_53_0\stage\lib\boost_python-mt-py26.dll.a')

def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


def git_version():
    ''' Return the sha1 of local git HEAD as a string. '''
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
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "unknown-git"
    return git_revision

def write_text(filename, text, mode='w'):
    with open(filename, mode='w') as a:
        try:
            a.write(text)
        except Exception as e:
            print(e)

def write_version_py(filename=os.path.join('hotspotter', 'generated_version.py')):
    cnt = """ # THIS FILE IS GENERATED FROM HOTSPOTTER SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
git_revision = '%(git_revision)s'
full_version = '%(version)s.dev-%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULL_VERSION = VERSION
    if os.path.isdir('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        GIT_REVISION = "RELEASE"
    else:
        GIT_REVISION = "unknown-git"

    FULL_VERSION += '.dev-' + GIT_REVISION
    text = cnt % {'version': VERSION,
                  'full_version': FULL_VERSION,
                  'git_revision': GIT_REVISION,
                  'isrelease': str(ISRELEASED)}
    write_text(filename, text)

def do_setup():
    write_version_py()
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          license=LICENSE,
          platforms=PLATFORMS,
          packages=find_packages(),
          install_requires=INSTALL_REQUIRES,
          install_optional=INSTALL_OPTIONAL,
          build_scripts = ['hotspotter/scripts/compile_widgets.bat'],
          keywords=' '.join([
            'hotsoptter', 'vision', 'animals', 'object recognition',
            'instance recognition', 'naive bayes' ]),
          cmdclass = {'build_py': build_py,
                      'build_scripts': build_scripts}
    )

if __name__ == "__main__":
    import sys
    print 'Entering HotSpotter setup'
    if len(sys.argv) <= 1:
        print 'what should I do? setup_windows? do_setup? install?'
    for cmd in iter(sys.argv[1:]):
        if cmd == 'setup_boost':
            setup_boost()
    #do_setup()

