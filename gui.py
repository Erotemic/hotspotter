from __future__ import division, print_function
import __builtin__
import fileio as io
import os
import os.path
import sys

import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    print('gui> Configuring matplotlib for Qt4')
    matplotlib.use('Qt4Agg')

import PyQt4.Qt
import PyQt4.QtCore as QtCore
from PyQt4.Qt import QMessageBox
from PyQt4.Qt import QCoreApplication, QApplication
from PyQt4.Qt import QMainWindow, QTableWidgetItem, QMessageBox, \
        QAbstractItemView,  QWidget, Qt, pyqtSlot, pyqtSignal, \
        QStandardItem, QStandardItemModel, QString, QObject

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[gui] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

IS_INIT = False

def make_dummy_main_window():
    mainwin = PyQt4.Qt.QMainWindow()
    mainwin.setWindowTitle('Dummy Main Window')
    mainwin.show()
    return mainwin
    

class HotspotterMainWindow(QMainWindow):
    def __init__(hsgui):
        super( HotspotterMainWindow, hsgui ).__init__()

def select_files(caption='Select Files:', directory=None):
    ''' EG: 
    image_list = select_files('Select one or more images to add.')
    print('Image List: '+repr(image_list))
    '''
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
    dpath = str(qdlg.getExistingDirectory(caption=caption, options=qopt, directory=directory))
    print('Selected Directory: '+dpath)
    io.global_cache_write('select_directory', os.path.split(dpath)[0])
    return dpath

@pyqtSlot(name='create_new_database')
def create_new_database():
    db_dir = gui.select_directory('Create a new directory to be used as the database')
    io.global_cache_write('db_dir', db_dir)
    select_database_dir(db_dir)

def select_database_dir(db_dir):
    do_convert = not exists(join(db_dir, '.hs_internals'))
    if do_convert:
        img_dpath = join(db_dir,'images')
        if not exists(img_dpath):
            img_dpath = gui.select_directory('Select directory with images in it')
        convert_hsdb.convert_named_chips(db_dir, img_dpath)
    
@pyqtSlot(name='open_old_database')
def open_old_database():
    db_dir = gui.select_directory('Select a database dir')
    io.global_cache_write('db_dir', db_dir)
    select_database_dir(db_dir)

def show_open_db_dlg(parent=None):
    from _frontend import OpenDatabaseDialog
    if not '-nc' in sys.argv and not '--nocache' in sys.argv: 
        db_dir = io.global_cache_read('db_dir')
        if db_dir == '.': 
            db_dir = None
    print('[gui] cached db_dir=%r' % db_dir)
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
    app = QCoreApplication.instance() 
    is_root = app is None
    if is_root: # if not in qtconsole
        print('Initializing QApplication')
        app = QApplication(sys.argv)
    else: 
        print('Parent already initialized QApplication')
    try:
        __IPYTHON__
        is_root = False
    except NameError as ex:
        # You are not root if you are in IPYTHON
        pass
    IS_INIT = True
    return app, is_root

def run_main_loop(app, is_root=True):
    if is_root:
        print('[gui] running main loop.')
        sys.exit(app.exec_())
    else:
        print('[gui] using roots main loop')

def msgbox(msg, title='msgbox'):
    'Make a non modal critical QMessageBox.'
    msgBox = QMessageBox(None);
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
