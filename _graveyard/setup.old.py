import textwrap
from os.path import isdir, isfile
from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
from _setup import git_helpers as git_helpers

def get_hotspotter_datafiles():
    'Build the data files used by py2exe and py2app'
    import matplotlib
    data_files = []
    # Include Matplotlib data (for figure images and things)
    data_files.extend(matplotlib.get_py2exe_datafiles())
    # Include TPL Libs
    plat_tpllibdir = join('hotspotter', '_tpl', 'lib', sys.platform)
    if sys.platform == 'win32':
        # Hack to get MinGW dlls in for FLANN
        data_files.append(('',[join(plat_tpllibdir, 'libgcc_s_dw2-1.dll'),
                               join(plat_tpllibdir,'libstdc++-6.dll')]))
    if sys.platform == 'darwin':
        pass
    else:
        for root,dlist,flist in os.walk(plat_tpllibdir):
            tpl_dest = root
            tpl_srcs = [realpath(join(root,fname)) for fname in flist]
            data_files.append((tpl_dest, tpl_srcs))
    # Include Splash Screen
    splash_dest = normpath('_frontend')
    splash_srcs = [realpath('_frontend/splash.png')]
    data_files.append((splash_dest, splash_srcs))
    return data_files


if cmd in ['localize', 'setup_localize.py']:
    from setup_localize import *
#package_application() - moved to graveyard (pyapp_pyexe.py)


def get_system_setup_kwargs():
    return dict(
        platforms=PLATFORMS,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        install_optional=INSTALL_OPTIONAL,
    )

def get_info_setup_kwarg():
    return dict(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=' '.join([
            'hotsoptter', 'vision', 'animals', 'object recognition',
            'instance recognition', 'naive bayes' ]))

def ensure_findable_windows_dlls():
    numpy_core = r'C:\Python27\Lib\site-packages\numpy\core'
    numpy_libs = ['libiomp5md.dll', 'libifcoremd.dll', 'libiompstubs5md.dll', 'libmmd.dll']
    pydll_dir  = r'C:\Python27\DLLs'
    for nplib in numpy_libs:
        dest = join(pydll_dir, nplib)
        if not exists(dest):
            src = join(numpy_core, nplib)
            shutil.copyfile(src, dest)
    zmqpyd_target = r'C:\Python27\DLLs\libzmq.pyd'
    if not exists(zmqpyd_target):
        #HACK http://stackoverflow.com/questions/14870825/
        #py2exe-error-libzmq-pyd-no-such-file-or-directory
        pyzmg_source = r'C:\Python27\Lib\site-packages\zmq\libzmq.pyd'
        shutil.copyfile(pyzmg_source, zmqpyd_target)



def write_text(filename, text, mode='w'):
    with open(filename, mode='w') as a:
        try:
            a.write(text)
        except Exception as e:
            print(e)

def write_version_py(filename=None):
    if filename is None:
        hsdir = os.path.split(realpath(__file__))[0]
        filename = join(hsdir, 'generated_version.py')
    cnt = textwrap.dedent('''
    # THIS FILE IS GENERATED FROM HOTSPOTTER SETUP.PY
    short_version = '%(version)s'
    version = '%(version)s'
    git_revision = '%(git_revision)s'
    full_version = '%(version)s.dev-%(git_revision)s'
    release = %(isrelease)s
    if not release:
        version = full_version''')
    FULL_VERSION = VERSION
    if isdir('.git'):
        GIT_REVISION = git_helpers.git_version()
    # must be a source distribution, use existing version file
    elif exists(filename):
        GIT_REVISION = 'RELEASE'
    else:
        GIT_REVISION = 'unknown-git'
    FULL_VERSION += '.dev-' + GIT_REVISION
    text = cnt % {'version': VERSION,
                  'full_version': FULL_VERSION,
                  'git_revision': GIT_REVISION,
                  'isrelease': str(ISRELEASED)}
    write_text(filename, text)




def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = join(where,name)
            if ('.' not in name and isdir(fn) and
                isfile(join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out

if cmd == 'setup_boost':
    setup_boost()
    sys.exit(0)
