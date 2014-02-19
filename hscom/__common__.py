from __future__ import division, print_function
import __builtin__
import sys

__QUIET__ = '--quiet' in sys.argv
__AGGROFLUSH__ = '--aggroflush' in sys.argv
__DEBUG__ = False

__MODULE_LIST__ = []

try:
    profile  # NoQA
    if not __QUIET__:
        __builtin__.print('[common] profiling with kernprof.')
except NameError:
    if not __QUIET__:
        __builtin__.print('[common] not profiling.')
    profile = lambda func: func


def get_modules():
    return __MODULE_LIST__


def init(module_name, module_prefix='[???]', DEBUG=None, initmpl=False):
    # implicitly imports a set of standard functions into hotspotter modules
    # makes keeping track of printing much easier
    global __MODULE_LIST__
    module = sys.modules[module_name]
    __MODULE_LIST__.append(module)

    if __DEBUG__:
        __builtin__.print('[common] import %s  # %s' % (module_name, module_prefix))

    # Define reloading function
    def rrr():
        'Dynamic module reloading'
        global __DEBUG__
        import imp
        prev = __DEBUG__
        __DEBUG__ = False
        print(module_prefix + ' reloading ' + module_name)
        imp.reload(module)
        __DEBUG__ = prev

    if __DEBUG__:
        # Extra debug info
        def print(msg):
            __builtin__.print(module_prefix + str(msg).replace(module_prefix, ''))

        def print_(msg):
            sys.stdout.write(module_prefix + str(msg).replace(module_prefix, ''))
            sys.stdout.flush()
    else:
        if __AGGROFLUSH__:
            def print_(msg):
                sys.stdout.write(msg)
                sys.stdout.flush()
        else:
            # Define a print that doesnt flush
            def print_(msg):
                sys.stdout.write(msg)
        # Define a print that flushes
        def print(msg):
            __builtin__.print(msg)

    # Function to to turn printing off
    def noprint(msg):
        pass

    # Closures are cool
    def print_on():
        module.print = print
        module.print_ = print_

    def print_off():
        module.print = noprint
        module.print_ = noprint

    if DEBUG is None:
        return print, print_, print_on, print_off, rrr, profile

    # Define a printdebug function
    if DEBUG:
        def printDBG(msg):
            print(module_prefix + ' DEBUG ' + msg)
    else:
        def printDBG(msg):
            pass

    # Initialize matplotlib if requested
    if initmpl:
        import matplotlib
        import multiprocessing
        backend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            if not __QUIET__:
                print('[common] ' + module_prefix + ' current backend is: %r' % backend)
                print('[common] ' + module_prefix + ' matplotlib.use(Qt4Agg)')
            if backend != 'Qt4Agg':
                matplotlib.use('Qt4Agg', warn=True, force=True)
                backend = matplotlib.get_backend()
                print(module_prefix + ' current backend is: %r' % backend)
            if '--notoolbar' in sys.argv or '--devmode' in sys.argv:
                toolbar = 'None'
            else:
                toolbar = 'toolbar2'
            matplotlib.rcParams['toolbar'] = toolbar
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
