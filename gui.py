import matplotlib
print('Configuring matplotlib for Qt4')
matplotlib.use('Qt4Agg')

import PyQt4.Qt
import PyQt4.QtCore as QtCore
from PyQt4.Qt import QMessageBox
from PyQt4.Qt import QCoreApplication, QApplication
from PyQt4.Qt import QMainWindow, QTableWidgetItem, QMessageBox, \
        QAbstractItemView,  QWidget, Qt, pyqtSlot, pyqtSignal, \
        QStandardItem, QStandardItemModel, QString, QObject

import fileio as io
import os
import os.path

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

QT_IS_INIT = False

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

def init_qtapp():
    global QT_IS_INIT
    app = QCoreApplication.instance() 
    is_root = app is None
    if is_root: # if not in qtconsole
        print('Initializing QApplication')
        app = QApplication(sys.argv)
    else: 
        print('Parent already initialized QApplication')
    QT_IS_INIT = True
    return app, is_root

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
