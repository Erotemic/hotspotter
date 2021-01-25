
import builtins
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
    # TODO Merge util's argv stuff here or merge this there?
    # Or split it into sepearate top-level module?
    if name.find('--') == 0:
        name = name[2:]
    if '--' + name in sys.argv and default is False:
        return True
    if '--no' + name in sys.argv and default is True:
        return False
    return default

__QUIET__      = argv_flag('--quiet', False)
__AGGROFLUSH__ = argv_flag('--aggroflush', False)
__DEBUG__      = argv_flag('--debug', False)
__INDENT__     = argv_flag('--indent', True)
__LOGGING__    = argv_flag('--logging', True)


log_fname = 'hotspotter_logs_%04d.out'
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
        builtins.print('[common] profiling with kernprof.')
except NameError:
    if __DEBUG__:
        builtins.print('[common] not profiling.')
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
__STDOUT__ = sys.stdout
HS_PRINT_FUNCTION = builtins.print
HS_DBG_PRINT_FUNCTION = builtins.print
HS_WRITE_FUNCTION = __STDOUT__.write
HS_FLUSH_FUNCTION = __STDOUT__.flush


def add_logging_handler(handler, default_format=True):
    global root_logger
    if default_format:
        logformat = '[%(asctime)s]%(message)s'
        timeformat = '%H:%M:%S'
        # create formatter and add it to the handlers
        #logformat = '%Y-%m-%d %H:%M:%S'
        #logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(logformat, timeformat)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def create_logger():
    global root_logger
    global HS_PRINT_FUNCTION
    global HS_DBG_PRINT_FUNCTION
    global HS_WRITE_FUNCTION
    if root_logger is None:
        #logging.config.dictConfig(LOGGING)
        msg = ('logging to log_fpath=%r' % log_fpath)
        HS_PRINT_FUNCTION(msg)
        root_logger = logging.getLogger('root')
        root_logger.setLevel('DEBUG')
        # create file handler which logs even debug messages
        #fh = logging.handlers.WatchedFileHandler(log_fpath)
        fh = logging.FileHandler(log_fpath)
        ch = logging.StreamHandler(__STDOUT__)
        add_logging_handler(fh)
        add_logging_handler(ch)
        root_logger.propagate = False
        root_logger.setLevel(logging.DEBUG)
        # print success
        HS_WRITE_FUNCTION = lambda msg: root_logger.info(msg)
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
        builtins.print('[common] import %s  # %s' % (module_name, module_prefix))

    # Define reloading function
    def rrr():
        'Dynamic module reloading'
        global __DEBUG__
        import imp
        prev = __DEBUG__
        __DEBUG__ = False
        builtins.print(module_prefix + ' reloading ' + module_name)
        imp.reload(module)
        __DEBUG__ = prev

    # Define log_print
    if __DEBUG__:
        def log_print(msg):
            HS_PRINT_FUNCTION(module_prefix + str(msg))

        def log_print_(msg):
            HS_WRITE_FUNCTION(module_prefix + str(msg))
    else:
        if __AGGROFLUSH__:
            def log_print_(msg):
                HS_WRITE_FUNCTION(msg)
                HS_FLUSH_FUNCTION()
        else:
            def log_print_(msg):
                HS_WRITE_FUNCTION(msg)
        def log_print(msg):
            HS_PRINT_FUNCTION(msg)

    def noprint(msg):
        #logger.debug(module_prefix + ' DEBUG ' + msg)
        pass

    # Define print switches
    # Closures are cool
    def print_on():
        if not module in __MODULE_LIST__:
            __MODULE_LIST__.append(module)  # SO HACKY
        module.print = log_print
        module.print_ = log_print_

    def print_off():
        if module in __MODULE_LIST__:
            __MODULE_LIST__.remove(module)  # SO HACKY
        module.print = noprint
        module.print_ = noprint

    # ACTUALLY SET PRINT:
    # FIXME we dont actually have to overwrite the name in this module
    print  = log_print
    print_ = log_print_

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
            mpl_keypress_shortcuts = [key for key in list(matplotlib.rcParams.keys()) if key.find('keymap') == 0]
            for key in mpl_keypress_shortcuts:
                matplotlib.rcParams[key] = ''
            #matplotlib.rcParams['text'].usetex = False
            #for key in mpl_keypress_shortcuts:
                #print('%s = %s' % (key, matplotlib.rcParams[key]))
            # Disable mpl shortcuts
                #matplotlib.rcParams['toolbar'] = 'None'
                #matplotlib.rcParams['interactive'] = True

    return print, print_, print_on, print_off, rrr, profile, printDBG
