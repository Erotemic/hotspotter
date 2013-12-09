from __future__ import division, print_function
import __builtin__
from os.path import split
import sys
import PyQt4
from PyQt4 import Qt, QtCore, QtGui
import fileio as io

IS_INIT = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[guitools] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

IS_INIT = False

def select_files(caption='Select Files:', directory=None):
    'Selects one or more files from disk using a qt dialog'
    print(caption)
    if directory is None: 
        directory = io.global_cache_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory)
    file_list = map(str, qfile_list)
    print('Selected Files: '+str(file_list))
    io.global_cache_write('select_directory', directory)
    return file_list

def select_directory(caption='Select Directory', directory=None):
    print(caption)
    if directory is None: 
        directory = io.global_cache_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qopt = PyQt4.Qt.QFileDialog.ShowDirsOnly
    qdlg_kwargs = dict(caption=caption, options=qopt, directory=directory)
    dpath = str(qdlg.getExistingDirectory(**qdlg_kwargs))
    print('Selected Directory: '+dpath)
    io.global_cache_write('select_directory', split(dpath)[0])
    return dpath
    
def show_open_db_dlg(parent=None):
    from _frontend import OpenDatabaseDialog
    if not '-nc' in sys.argv and not '--nocache' in sys.argv: 
        db_dir = io.global_cache_read('db_dir')
        if db_dir == '.': 
            db_dir = None
    print('[guitools] cached db_dir=%r' % db_dir)
    if parent is None:
        parent = PyQt4.QtGui.QDialog()
    opendb_ui = OpenDatabaseDialog.Ui_Dialog()
    opendb_ui.setupUi(parent)
    opendb_ui.new_db_but.clicked.connect(create_new_database)
    opendb_ui.open_db_but.clicked.connect(open_old_database)
    parent.show()
    return opendb_ui, parent

def init_qtapp():
    global IS_INIT
    app = Qt.QCoreApplication.instance() 
    is_root = app is None
    if is_root: # if not in qtconsole
        print('[guitools] Initializing QApplication')
        app = Qt.QApplication(sys.argv)
    try:
        __IPYTHON__
        is_root = False
    # You are not root if you are in IPYTHON
    except NameError as ex:
        pass
    IS_INIT = True
    return app, is_root

def run_main_loop(app, is_root=True, backend=None):
    if backend is not None:
        print('[guitools] setting active window')
        app.setActiveWindow(backend.win)
    if is_root:
        print('[guitools] running main loop.')
        sys.exit(app.exec_())
    else:
        print('[guitools] using roots main loop')

def msgbox(msg, title='msgbox'):
    'Make a non modal critical Qt.QMessageBox.'
    msgBox = Qt.QMessageBox(None);
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(Qt.QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox

def make_dummy_main_window():
    mainwin = PyQt4.Qt.QMainWindow()
    mainwin.setWindowTitle('Dummy Main Window')
    mainwin.show()
    return mainwin

def make_main_window(hs=None):
    import guiback
    backend = guiback.MainWindowBackend(hs=hs)
    backend.win.show()
    backend.clear_selection()
    return backend

def popup_menu(widget, opt2_callback):
    def popup_slot(pos):
        print(pos)
        menu = QtGui.QMenu()
        actions = [menu.addAction(opt, func) for opt, func in
                   iter(opt2_callback)]
        #pos=QtGui.QCursor.pos()
        selection = menu.exec_(widget.mapToGlobal(pos))
    return popup_slot

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    def test():
        app, is_root = init_qtapp()
        main_win = make_main_window()
        run_main_loop(app, is_root, main_win)
    test()
