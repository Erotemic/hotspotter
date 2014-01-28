from __future__ import division, print_function
import __builtin__
import sys

try:
    profile  # NoQA
    __builtin__.print('[common] profiling with kernprof.')
except NameError:
    __builtin__.print('[common] not profiling.')
    profile = lambda func: func


__DEBUG__ = False


def init(module_name, module_prefix='[???]', DEBUG=None, initmpl=False):
    module = sys.modules[module_name]
    aggroflush = '--aggroflush' in sys.argv

    if __DEBUG__:
        __builtin__.print('[common] import %s  # %s' % (module_name, module_prefix))

    def rrr():
        'Dynamic module reloading'
        global __DEBUG__
        import imp
        prev = __DEBUG__
        __DEBUG__ = False
        print(module_prefix + ' reloading ' + module_name)
        imp.reload(module)
        __DEBUG__ = prev

    if aggroflush:
        def print_(msg):
            sys.stdout.write(msg)
            sys.stdout.flush()
    else:
        def print_(msg):
            sys.stdout.write(msg)

    def print(msg):
        __builtin__.print(msg)
        #__builtin__.print(module_prefix + msg)

    def noprint(msg):
        pass

    def print_on():
        module.print = print
        module.print_ = print_

    def print_off():
        module.print = noprint
        module.print_ = noprint

    if DEBUG is None:
        return print, print_, print_on, print_off, rrr, profile

    if DEBUG:
        def printDBG(msg):
            print(module_prefix + ' DEBUG ' + msg)
    else:
        def printDBG(msg):
            pass

    if initmpl:
        import matplotlib
        import multiprocessing
        backend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            print('[common] ' + module_prefix + ' current backend is: %r' % backend)
            print('[common] ' + module_prefix + ' matplotlib.use(Qt4Agg)')
            if backend != 'Qt4Agg':
                matplotlib.use('Qt4Agg', warn=True, force=True)
                backend = matplotlib.get_backend()
                print(module_prefix + ' current backend is: %r' % backend)
            matplotlib.rcParams['toolbar'] = 'toolbar2'
            matplotlib.rc('text', usetex=False)
            mpl_keypress_shortcuts = [key for key in matplotlib.rcParams.keys() if key.find('keymap') == 0]
            for key in mpl_keypress_shortcuts:
                matplotlib.rcParams[key] = ''
            #matplotlib.rcParams['text'].usetex = False
            #for key in mpl_keypress_shortcuts:
                #print('%s = %s' % (key, matplotlib.rcParams[key]))
            # Disable mpl shortcuts
                #matplotlib.rcParams['toolbar'] = 'None'
                #matplotlib.rcParams['interactive'] = True

    return print, print_, print_on, print_off, rrr, profile, printDBG
