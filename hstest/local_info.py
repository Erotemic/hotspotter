from __future__ import division, print_function
# Disable common logging
import sys
sys.argv.extend(['--nologging', '--noindent'])
import os  # NOQA
from os.path import join, expanduser, exists, split  # NOQA
from hscom import helpers as util
import psutil  # NOQA


# Import using debug import
def ensure_hotspotter():
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    sys.path.append(hotspotter_dir)
ensure_hotspotter()
from dbgimport import *  # NOQA


util.printvar2('cv2.__version__')
util.printvar2('multiprocessing.cpu_count()')
util.printvar2('sys.platform')
util.printvar2('os.getcwd()')

print('')
print('Python Site')
print('')

util.printvar2('site.getsitepackages()')
util.printvar2('site.getusersitepackages()')
print('')
print('PSUTIL CPUS')
print('')
util.printvar2('psutil.cpu_times()')
util.printvar2('psutil.NUM_CPUS')
print('')
print('PSUTIL MEMORY')
print('')
util.printvar2('psutil.virtual_memory()')
util.printvar2('psutil.swap_memory()')
print('')
print('PSUTIL DISK')
print('')
util.printvar2('psutil.disk_partitions()')
util.printvar2('psutil.disk_usage("/")')
util.printvar2('psutil.disk_io_counters()')
print('')
print('PSUTIL NETWORK')
print('')
util.printvar2('psutil.net_io_counters(pernic=True)')
print('')
print('PSUTIL MISC')
print('')
util.printvar2('psutil.get_users()')
util.printvar2('psutil.get_boot_time()')
util.printvar2('psutil.get_pid_list()')


psutil.test()


import resource
util.rrr()
used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print('[parallel] Max memory usage: %s' % util.byte_str2(used_memory))


if '--cmd' in sys.argv:
    util.embed()
