from __future__ import print_function, division
import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[parallel]', DEBUG=False)
# Python
import psutil
import os
# HotSpotter
import helpers as util


def peak_memory():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def current_memory_usage():
    meminfo = psutil.Process(os.getpid()).get_memory_info()
    rss = meminfo[0]  # Resident Set Size / Mem Usage
    vms = meminfo[1]  # Virtual Memory Size / VM Size  # NOQA
    return rss


def num_cpus():
    return psutil.NUM_CPUS


def available_memory():
    return psutil.virtual_memory().available


def total_memory():
    return psutil.virtual_memory().total


def used_memory():
    return total_memory() - available_memory()


def memstats():
    print('total = %s' % util.byte_str2(total_memory()))
    print('available = %s' % util.byte_str2(available_memory()))
    print('used = %s' % util.byte_str2(used_memory()))
    print('current = %s' % util.byte_str2(current_memory_usage()))

if __name__ == '__main__':
    memstats()


#psutil.virtual_memory()
#psutil.swap_memory()
#psutil.disk_partitions()
#psutil.disk_usage("/")
#psutil.disk_io_counters()
#psutil.net_io_counters(pernic=True)
#psutil.get_users()
#psutil.get_boot_time()
#psutil.get_pid_list()
