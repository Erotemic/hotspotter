import subprocess
import sys
import os
import struct

#win32api.SetFileAttributes(path, win32con.FILE_ATTRIBUTE_NORMAL)

def is_32_bit_python():
    return struct.calcsize("P")*8 == 32
def is_windows():
    return sys.platform == 'win32'
def is_linux():
    return sys.platform == 'linux2'
def is_mac():
    return sys.platform == 'darwin'

def view_text_file(txtfile):
    if sys.platform == 'win32':
        os.startfile(txtfile)
    elif sys.platform == 'linux2':
        subprocess.Popen('gedit '+txtfile, shell=True)
    elif sys.platform == 'darwin':
        subprocess.Popen('open '+txtfile, shell=True)
    else:
        raise Exception('Unknown platform: '+str(sys.platform))
#---
def view_directory(dname):
    os_type       = sys.platform
    open_prog_map = {'win32':'explorer.exe', 'linux2':'nautilus', 'darwin':'open'}
    open_prog     = open_prog_map[os_type]
    os.system(open_prog+' '+dname)

def platexec(exe_fpath):
    if sys.platform == 'win32':
        return '"'+os.path.normpath(exe_fpath)+'.exe"'
    if sys.platform == 'linux2':
        return '"'+os.path.normpath(exe_fpath)+'.ln"'
    if sys.platform == 'darwin':
        return '"'+os.path.normpath(exe_fpath)+'.mac"'

    #toreturn = os.path.normpath(path).replace("'","").replace('"','')
    #if quotes_bit:
    #   toreturn = '"'+toreturn+'"'
    #return toreturn
