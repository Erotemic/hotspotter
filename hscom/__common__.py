from __future__ import division, print_function
import __builtin__
from os.path import exists, join
import logging
import logging.config
import os
import sys
import multiprocessing


__MODULE_LIST__ = []
__LOGGERS__ = {}
__IN_MAIN_PROCESS__ = multiprocessing.current_process().name == 'MainProcess'


def argv_flag(name, default):
    if '--' + name in sys.argv and default is False:
        return True
    if '--no' + name in sys.argv and default is True:
        return False
    return default

__QUIET__ = argv_flag('quiet', False)
__AGGROFLUSH__ = argv_flag('aggroflush', False)
__DEBUG__ = argv_flag('debug', False)
__INDENT__ = argv_flag('indent', True)
__LOGGING__ = argv_flag('logging', True)


log_fname = 'hotspotter_logs_%d.out'
log_dir = 'logs'

if not exists(log_dir):
    os.makedirs(log_dir)

count = 0
log_fpath = join(log_dir, log_fname % count)
while exists(log_fpath):
    log_fpath = join(log_dir, log_fname % count)
    count += 1

try:
    profile  # NoQA
    if __DEBUG__:
        __builtin__.print('[common] profiling with kernprof.')
except NameError:
    if __DEBUG__:
        __builtin__.print('[common] not profiling.')
    profile = lambda func: func

 #|  %(name)s            Name of the logger (logging channel)
 #|  %(levelno)s         Numeric logging level for the message (DEBUG, INFO,
 #|                      WARNING, ERROR, CRITICAL)
 #|  %(levelname)s       Text logging level for the message ("DEBUG", "INFO",
 #|                      "WARNING", "ERROR", "CRITICAL")
 #|  %(pathname)s        Full pathname of the source file where the logging
 #|                      call was issued (if available)
 #|  %(filename)s        Filename portion of pathname
 #|  %(module)s          Module (name portion of filename)
 #|  %(lineno)d          Source line number where the logging call was issued
 #|                      (if available)
 #|  %(funcName)s        Function name
 #|  %(created)f         Time when the LogRecord was created (time.time()
 #|                      return value)
 #|  %(asctime)s         Textual time when the LogRecord was created
 #|  %(msecs)d           Millisecond portion of the creation time
 #|  %(relativeCreated)d Time in milliseconds when the LogRecord was created,
 #|                      relative to the time the logging module was loaded
 #|                      (typically at application startup time)
 #|  %(thread)d          Thread ID (if available)
 #|  %(threadName)s      Thread name (if available)
 #|  %(process)d         Process ID (if available)
 #|  %(message)s         The result of record.getMessage(), computed just as
 #|                      the record is emitted

root_logger = None
HS_PRINT_FUNCTION = __builtin__.print
HS_DBG_PRINT_FUNCTION = __builtin__.print
HS_WRITE_FUNCTION = sys.stdout.write
HS_FLUSH_FUNCTION = sys.stdout.flush


def create_logger():
    global root_logger
    global HS_PRINT_FUNCTION
    global HS_DBG_PRINT_FUNCTION
    if root_logger is None:
        #logging.config.dictConfig(LOGGING)
        msg = ('logging to log_fpath=%r' % log_fpath)
        HS_PRINT_FUNCTION(msg)
        root_logger = logging.getLogger('root')
        root_logger.setLevel('DEBUG')
        # create file handler which logs even debug messages
        #fh = logging.handlers.WatchedFileHandler(log_fpath)
        fh = logging.FileHandler(log_fpath)
        ch = logging.StreamHandler()
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        #logformat = '%Y-%m-%d %H:%M:%S'
        #logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logformat = '[%(asctime)s]%(message)s'
        timeformat = '%H:%M:%S'
        formatter = logging.Formatter(logformat, timeformat)
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        root_logger.addHandler(ch)
        root_logger.addHandler(fh)
        root_logger.propagate = False
        # print success
        HS_PRINT_FUNCTION = lambda msg: root_logger.info(msg)
        HS_DBG_PRINT_FUNCTION = lambda msg: root_logger.debug(msg)
        HS_PRINT_FUNCTION('logger init')
        HS_PRINT_FUNCTION(msg)


def get_modules():
    if __INDENT__:
        return __MODULE_LIST__
    else:
        return []


def init(module_name, module_prefix='[???]', DEBUG=None, initmpl=False):
    global root_logger
    # implicitly imports a set of standard functions into hotspotter modules
    # makes keeping track of printing much easier
    global __MODULE_LIST__
    module = sys.modules[module_name]
    __MODULE_LIST__.append(module)
    if __IN_MAIN_PROCESS__ and __LOGGING__:
        create_logger()

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
            HS_PRINT_FUNCTION(module_prefix + str(msg))

        def print_(msg):
            HS_WRITE_FUNCTION(module_prefix + str(msg))
    else:
        if __AGGROFLUSH__:
            def print_(msg):
                HS_WRITE_FUNCTION(msg)
                HS_FLUSH_FUNCTION()
        else:
            # Define a print that doesnt flush
            def print_(msg):
                HS_WRITE_FUNCTION(msg)
        # Define a print that flushes
        def print(msg):
            HS_PRINT_FUNCTION(msg)

    # Function to to turn printing off
    def noprint(msg):
        #logger.debug(module_prefix + ' DEBUG ' + msg)
        pass

    # Closures are cool
    def print_on():
        module.print = print
        module.print_ = print_

    def print_off():
        module.print = print
        module.print = noprint
        module.print_ = noprint

    if DEBUG is None:
        return print, print_, print_on, print_off, rrr, profile

    # Define a printdebug function
    if DEBUG:
        def printDBG(msg):
            HS_DBG_PRINT_FUNCTION(module_prefix + ' DEBUG ' + msg)
    else:
        def printDBG(msg):
            #logger.debug(module_prefix + ' DEBUG ' + msg)
            pass

    # Initialize matplotlib if requested
    if initmpl:
        import matplotlib
        backend = matplotlib.get_backend()
        if __IN_MAIN_PROCESS__:
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
