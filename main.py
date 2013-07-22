#!/usr/bin/env python

# For py2exe
import PIL.TiffImagePlugin
import PIL.Image
import PIL.PngImagePlugin
import PIL.JpegImagePlugin
import PIL.GifImagePlugin
import PIL.PpmImagePlugin

import argparse
import inspect
import os, sys
from os.path import join, dirname

def emergency_msgbox(title, msg):
    'Make a non modal critical QMessageBox.'
    from PyQt4.Qt import QMessageBox
    msgBox = QMessageBox(None);
    msgBox.setAttribute(Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox

def ensure_tpl_libs():
    print('Ensuring third party libraries')
    try: # Ensure that TPL's lib files are in PATH
        #from hotspotter.standalone import find_hotspotter_root_dir
        print('Can import hotspotter?')
        import hotspotter
        print(' ... yes')
        TPL_LIB_DIR = join(dirname(hotspotter.__file__), 'tpl/lib', sys.platform)
        sys.path.insert(0, TPL_LIB_DIR)
        ext = {'linux2':'.ln','darwin':'.mac','win32':'.exe'}[sys.platform]
        # Ensure that hesaff is executable
        hesaff_fname = TPL_LIB_DIR+'/hesaff'+ext
        is_executable = lambda fname: bin(int(oct(os.stat(fname).st_mode)[4]))[4]
        if not is_executable(hesaff_fname): 
            os.system('chmod 775 '+hesaff_fname)
        print('Can import cv2?')
        import cv2
        print(' ... yes')
        print('Can import hotspotter.tpl.pyflann?')
        import hotspotter.tpl.pyflann
        print(' ... yes')
    except Exception as ex: 
        print('\n\n!!! TPL ERROR !!!')
        PYTHONPATH = os.getenv('PYTHONPATH')
        PATH = os.getenv('PATH')

        print('PYTHONPATH = '+repr(PYTHONPATH))
        print('PATH = '+repr(PATH))

        print('''You must download hotspotter\'s 3rd party libraries before you can run it. 
        git clone https://github.com/Erotemic:tpl-hotspotter.git tpl''')
        raise

def parse_arguments():
    print('Parsing arguments')
    parser = argparse.ArgumentParser(description='HotSpotter - Instance Recognition', prefix_chars='+-')
    def_on  = {'action':'store_false', 'default':True}
    def_off = {'action':'store_true', 'default':False}
    parser.add_argument('-l', '--log-all', 
                        dest='logall_bit', help='Writes all logs', **def_off)
    parser.add_argument('--cmd', dest='cmd_bit',
                        help='Forces command line mode', **def_off)
    parser.add_argument('-g', '--gui-off', dest='gui_bit',
                        help='Runs HotSpotter in command line mode', **def_on)
    parser.add_argument('-a', '--autoload-off', dest='autoload_bit',
                        help='Starts HotSpotter without loading a database', **def_on)
    parser.add_argument('-dp', '--delete-preferences', dest='delpref_bit',
                        help='Deletes preferences in ~/.hotspotter', **def_off)
    args, unknown = parser.parse_known_args()
    return args


def initQtApp():
    # Attach to QtConsole's QApplication if able
    from PyQt4.Qt import QCoreApplication, QApplication
    app = QCoreApplication.instance() 
    isRootApp = app is None
    if isRootApp: # if not in qtconsole
        # configure matplotlib 
        import matplotlib
        print('Configuring matplotlib for Qt4')
        matplotlib.use('Qt4Agg')
        # Run new root application
        print('Starting new QApplication')
        app = QApplication(sys.argv)
    else: 
        print('Running using parent QApplication')
    return app, isRootApp 

def executeEventLoop(app):
    print('Running the application event loop')
    sys.stdout.flush()
    sys.exit(app.exec_())

# MAIN ENTRY POINT
if __name__ == '__main__':
    # 1) Multiprocess Initialization 
    from multiprocessing import freeze_support
    freeze_support()
    # 2) TPL Initialization
    ensure_tpl_libs()
    # 3) Qt Initialization
    args = parse_arguments()
    app, isRootApp = initQtApp()
    # 4) HotSpotter Initialization
    from hotspotter.other.logger import hsl

    from hotspotter.standalone import delete_preference_dir
    from hotspotter.Facade import Facade
    if args.logall_bit:
        hsl.enable_global_logs()
    if args.delpref_bit:
        delete_preference_dir()
    # 5) HotSpotter Execution 
    fac = Facade(use_gui=args.gui_bit, autoload=args.autoload_bit)

    # Register Facade functions into current namespace
    # ### SNIPIT: Namespace Class Functions
    for (name, value) in inspect.getmembers(Facade, predicate=inspect.ismethod):
        if name.find('_') != 0:
            exec('def '+name+'(*args, **kwargs): fac.'+name+'(*args, **kwargs)')
    # ### ---
    # Defined Aliases
    stat, status   = [lambda          : fac.print_status()]*2
    removec,       = [lambda          : fac.remove_cid()]
    rename,        = [lambda new_name : fac.rename_cid(new_name)]

    # Get developer variables
    # ### SNIPIT: Execute File
    with open('dev.py', 'r') as devfile:
        devpy = devfile.read()
        exec(devpy)
    # ### ----

    run_exec = isRootApp
    if args.cmd_bit:
        # Start IPython command line mode
        from hotspotter.helpers import in_IPython, have_IPython
        run_exec = False
        if not in_IPython() and have_IPython():
            import IPython
            IPython.embed()

    # Run Event Loop, but do not block QTConsole or IPython
    if run_exec:
        executeEventLoop(app)
