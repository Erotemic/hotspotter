from __init__ import *

import convert_to_hotspotterdb

import PyQt4.Qt
import PyQt4.QtCore as QtCore
from PyQt4.Qt import QMessageBox
from PyQt4.Qt import QCoreApplication, QApplication
from PyQt4.Qt import QMainWindow, QTableWidgetItem, QMessageBox, \
        QAbstractItemView,  QWidget, Qt, pyqtSlot, pyqtSignal, \
        QStandardItem, QStandardItemModel, QString, QObject
    
def config_matplotlib():
    # configure matplotlib 
    import matplotlib
    print('Configuring matplotlib for Qt4')
    matplotlib.use('Qt4Agg')

def init_qtapp():
    app = QCoreApplication.instance() 
    is_root = app is None
    if is_root: # if not in qtconsole
        print('Initializing QApplication')
        app = QApplication(sys.argv)
    else: 
        print('Parent already initialized QApplication')
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

TMP_DIR = realpath('.hotspotter/tmp')
helpers.ensuredir(TMP_DIR)

LASTDIR_FNAME = join(TMP_DIR, 'lastdir.txt')
LASTDBDIR_FNAME = join(TMP_DIR, 'lastdir.txt')

def cache_dir_read(cache_id):
    cache_fname = join(TMP_DIR, 'cached_dir_'+str(cache_id)+'.txt')
    return helpers.read_from(cache_fname) if exists(cache_fname) else '.'

def cache_dir_write(cache_id, newdir):
    cache_fname = join(TMP_DIR, 'cached_dir_'+str(cache_id)+'.txt')
    helpers.write_to(cache_fname, newdir)

def select_files(caption='Select Files:', directory=None):
    ''' EG: 
    image_list = select_files('Select one or more images to add.')
    print('Image List: '+repr(image_list))
    '''
    print(caption)
    qdlg = PyQt4.Qt.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory)
    file_list = map(str, qfile_list)
    print('Selected Files: '+dpath)
    return file_list

def select_directory(caption='Select Directory', directory=None):
    print(caption)
    if directory is None: 
        directory = cache_dir_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qopt = PyQt4.Qt.QFileDialog.ShowDirsOnly
    dpath = str(qdlg.getExistingDirectory(caption=caption, options=qopt, directory=directory))
    print('Selected Directory: '+dpath)
    cache_dir_write('select_directory', os.path.split(dpath)[0])
    return dpath
    
def catch_ctrl_c(signal, frame):
    print('Caught ctrl+c')

def signal_reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    signal.signal(signal.SIGINT, catch_ctrl_c)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('main> main.py')

    app, is_root = init_qtapp()
    signal_set()

    if True or '-wc' in sys.argv or '--withcache' in sys.argv:
        db_dir = cache_dir_read('db_dir')
        if db_dir == '.': 
            db_dir = None

    if db_dir is None or not exists(db_dir):
        db_dir = select_directory('Select a directory to be used as the database')
        cache_dir_write('db_dir', db_dir)

    do_convert = not exists(join(db_dir, '.hs_internals'))
    if do_convert:
        img_dpath = join(db_dir,'images')
        if not exists(img_dpath):
            img_dpath = select_directory('Select directory with images in it')
        convert_to_hotspotterdb.convert_named_chips(db_dir, img_dpath)


    hs = load_data2.HotSpotter(db_dir)

    if '--vrd' in sys.argv:
        helpers.vd(hs.dirs.result_dir)
        sys.exit(1)

    qcx2_res = mc2.run_matching(hs)

    allres = rr2.init_allres(hs, qcx2_res, SV=True)

    rr2.dump_all(allres)

    print('Exiting HotSpotter')
    sys.exit(0)
    #mainwin = PyQt4.Qt.QMainWindow()
    #mainwin.setWindowTitle('Dummy Main Window')
    #mainwin.show()
    #print('Running the application event loop')
    #helpers.flush()
    #sys.exit(app.exec_())

