from __future__ import division, print_function
import pip as pip_
import parse
from os.path import abspath, dirname
import subprocess
import sys


def pip(*args):
    pip_.main(list(args))


def install(package):
    pip_('install', package)


def upgrade(package):
    pip_('install', package)

core = [
    'Pygments>=1.6',
    'argparse>=1.2.1',
    'openpyxl>=1.6.2',  # reads excel xlsx files
    'parse>=1.6.2',
    'psutil>=1.0.1',
    'pyglet>=1.1.4',
    'pyparsing>=2.0.1',
    'pyreadline>=2.0',
    'python-dateutil>=1.5',
    'pyzmq>=13.1.0',  # distributed computing
    'six>=1.3.0',  # python 3 support
]

speed = [
    'Cython>=0.19.1',
    'pylru>=1.0.6',
    'llvmpy>=0.9.1',
    'numba>=0.3.2',
]

interface = [
    'ipython>=1.1.0',
    'matplotlib>=1.3.1',
    'python-qt>=0.50',
]

science = [
    'PIL>=1.1.7',
    'flann>=1.8.4',
    'numpy>=1.7.1',
    'opencv-python>=2.4.6',
    'pandas>=0.12.0',
    'scikit-image>=0.9.3',
    'scikit-learn>=0.14a1',
    'scipy>=0.12.0',
]

#File "<string>", line 3, in <module>
#File "C:\Python27\Lib\site-packages\matplotlib\backends\backend_webagg.py", line 19, in <module>
    #raise RuntimeError("The WebAgg backend requires Tornado.")

devtools = [
    'setuptools>=2.0.1'
    'distribute>='
    'pyinstaller>=2.1'
    'line-profiler>=1.0b3',
    'flake8>=2.1.0',
    'pep8>=1.4.6',
    'pyflakes>=0.7.3',
    'pylint>=1.0.0',
    'runsnakerun>=2.0.3',
    'squaremap>=1.0.2',
]

windows = [
    'winpexpect',
    'WinSys-3.x',
]

other = [
    'sympy',
    'supreme'  # super resolution
    'pytz',  # Timezones
    'grizzled',  # Utility library
    'Wand',  # ImageMagick
    'astroid',  # Syntax tree
    'boost-python',
    'colorama',  # ansii colors
    'mccabe',  # plugin for flake8
    'logilab-common',  # low level functions
    'nose',  # unit tester
]

allpkgs = core + speed + interface + science + devtools


def run_process(args, silent=True):
    PIPE = subprocess.PIPE
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
    if silent:
        (out, err) = proc.communicate()
    else:
        out_list = []
        for line in proc.stdout.readlines():
            print(line)
            sys.stdout.flush()
            out_list.append(line)
        out = '\n'.join(out_list)
        (_, err) = proc.communicate()
        ret = proc.wait()
    ret = proc.wait()
    return out, err, ret


def pipshow(pkg):
    out, err, ret = run_process('pip show ' + pkg)
    props_list = out.split('\r\n')[1:-1]
    props = dict([prop.split(': ') for prop in props_list])
    return props


def pipversion(pkg):
    return pipshow(pkg)['Version']


def pipinfo(pkg):
    out, err, ret = run_process('pip search ' + pkg)
    line_list = out.split('\r\n')
    pkginfolist = []
    next_entry = []
    for line in line_list:
        if line.find(' ') != 0:
            pkginfolist.append(''.join(next_entry))
            next_entry = []
        next_entry.append(line)

    found = []
    for pkginfo in pkginfolist:
        if pkginfo.find(pkg + ' ') == 0:

            def tryfmt1(pkginfo):
                parsestr1 = '{} - {} INSTALLED: {} LATEST: {}'
                name, desc, installed, latest = parse.parse(parsestr1, pkginfo)
                return name, desc, installed, latest

            def tryfmt2(pkginfo):
                parsestr2 = '{} - {} INSTALLED: {} (latest)'
                name, desc, installed = parse.parse(parsestr2, pkginfo)
                latest = installed
                return name, desc, installed, latest

            def tryfmt3(pkginfo):
                parsestr2 = '{} - {}'
                name, desc  = parse.parse(parsestr2, pkginfo)
                installed, latest = ('None', 'None')
                return name, desc, installed, latest

            for tryfmt in [tryfmt1, tryfmt2, tryfmt3]:
                try:
                    name, desc, installed, latest = tryfmt(pkginfo)
                    found.append(dict(pkg=name.strip(),
                                      info=desc.strip(),
                                      installed=installed.strip(),
                                      latest=latest.strip()))
                    break
                except TypeError:
                    pass

    if len(found) == 0:
        found = [dict(pkg=pkg, info='cannot find', installed=None, latest=None)]
    return found


def get_allpkg_info():
    allpkg_info = []
    for pkgstr in allpkgs:
        pkg, version = pkgstr.split('>=')
        info = pipinfo(pkg)
        tup = (info[0]['pkg'], info[0]['installed'], info[0]['latest'])
        print('pkg=%r installed=%r latest=%r' % (tup))
        allpkg_info.append(info)
    print_allpkg_info(allpkg_info)
    return allpkg_info


def print_allpkg_info(allpkg_info):
    for info in allpkg_info:
        tup = (info[0]['pkg'], info[0]['installed'], info[0]['latest'])
        buf =  ' ' * max(12 - len(info[0]['pkg']), 0)
        print(('%s ' + buf + 'installed=%r latest=%r') % tup)
#installed_list = pip.get_installed_distributions()
#install('runsnake')


PIP_DISABLE_ON_WINDOWS = [
    'wxPython',
    'PIL',
    'Pygments',
    'llvmpy',
    'matplotlib',
    'numba',
    'numpy',
    'python-qt',
    'pyzmg',
    'scipy',
]


#Get outdated packages
def get_outdated_packages(allpkg_info, safe=True):
    outdated = []
    unavailable = []
    for info in allpkg_info:
        pkg = info[0]['pkg']
        latest = info[0]['latest']
        installed = info[0]['installed']
        if sys.platform == 'win32' and safe and pkg in PIP_DISABLE_ON_WINDOWS:
            unavailable.append(info)
        elif installed is None or installed == 'None':
            unavailable.append(info)
        elif latest != installed:
            outdated.append(info)
    print('Pip does not seem to be managing:  \n    *' + '\n    *'.join([info[0]['pkg'] for info in unavailable]))
    print('Updates available for:  \n    *' + '\n    *'.join([info[0]['pkg'] + ' current=' + info[0]['installed'] + ' latest=' + info[0]['latest']for info in outdated]))
    return outdated, unavailable


def vd(path):
    'view directory'
    if sys.platform == 'win32':
        return run_process('explorer ' + path)


def write_installer_script(cmd_list, scriptname='installer'):
    if sys.platform != 'win32':
        ext = '.sh'
        cmd_list = ['sudo ' + cmd for cmd in cmd_list]
    else:
        ext = '.bat'
    script_fpath = abspath(scriptname + ext)
    with open(script_fpath, 'w') as file_:
        file_.write('\n'.join(cmd_list))
    vd(dirname(script_fpath))


def uninstall_windows_conflicts():
    # Uninstall windows things safely
    cmd_list = ['pip uninstall %s' % pkg for pkg in PIP_DISABLE_ON_WINDOWS]
    write_installer_script(cmd_list, scriptname='pip_uninstall')


if __name__ == '__main__':
    allpkg_info = get_allpkg_info()
    outdated, unavailable = get_outdated_packages(allpkg_info, False)

    cmd_list = ['pip install %s --upgrade' % info[0]['pkg'] for info in outdated]
    write_installer_script(cmd_list, scriptname='pip_upgrade')
    print('\n'.join(cmd_list))
