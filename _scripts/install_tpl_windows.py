import textwrap
import os
import sys

__FORCE_DOWNLOAD__ = '--force-download' in sys.argv

def str2list(multiline_str):
    return textwrap.dedent(multiline_str).strip().split('\n')

def wget(url):
    print 'Downloading: '+url
    cmd('wget '+url)

def cmd(cmd_):
    os.system(cmd_)

def install_from_url(url):
    url = url.replace('/download', '')
    _, exe_name = os.path.split(url)
    if not os.path.exists(exe_name) or __FORCE_DOWNLOAD__:
        wget(url)
    else:
        print exe_name+' has already been downloaded'
    print 'Running installer for: '+exe_name
    cmd(exe_name)
    
installer_urls = str2list('''
http://sourceforge.net/projects/scikit-learn/files/scikit-learn-0.13.1.win32-py2.7.exe
''')

#scikit-learn-0.13.1.win32-py2.7.exe

for url in iter(installer_urls):
    install_from_url(url)
