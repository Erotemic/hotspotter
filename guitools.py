from __future__ import division, print_function
import __builtin__
from os.path import split
import sys
import warnings
import numpy as np
import PyQt4
from PyQt4 import Qt, QtCore, QtGui
import fileio as io
import helpers

IS_INIT = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[*guitools] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

IS_INIT = False

def configure_matplotlib():
    import multiprocessing
    import matplotlib
    backend = matplotlib.get_backend()
    if multiprocessing.current_process().name == 'MainProcess':
        print('[*guitools] current backend is: %r' % backend)
        print('[*guitools] matplotlib.use(Qt4Agg)')
    else:
        return
    matplotlib.rcParams['toolbar'] = 'toolbar2'
    matplotlib.rc('text', usetex=False)
    #matplotlib.rcParams['text'].usetex = False
    if backend != 'Qt4Agg':
        matplotlib.use('Qt4Agg', warn=True, force=True)
        backend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            print('[*guitools] current backend is: %r' % backend)
        #matplotlib.rcParams['toolbar'] = 'None'
        #matplotlib.rcParams['interactive'] = True


# ---
def select_orientation():
    import draw_func2 as df2
    print('[*guitools] Define an orientation angle by clicking two points')
    try:
        # Compute an angle from user interaction
        sys.stdout.flush()
        fig = df2.gcf()
        pts = np.array(fig.ginput(2))
        #print('[*guitools] ginput(2) = %r' % pts)
        # Get reference point to origin 
        refpt = pts[0] - pts[1] 
        #theta = np.math.atan2(refpt[1], refpt[0])
        theta = np.math.atan(refpt[1]/refpt[0])
        logmsg('The angle in radians is: '+str(theta))
        return theta
    except Exception as ex: 
        logmsg('Annotate Orientation Failed'+str(ex))
        return None

# ---
def select_roi():
    import draw_func2 as df2
    from matplotlib.backend_bases import mplDeprecation
    print('[*guitools] Define a Rectanglular ROI by clicking two points.')
    try:
        sys.stdout.flush()
        fig = df2.gcf()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=mplDeprecation)
            pts = fig.ginput(2)
        print('[*guitools] ginput(2) = %r' % (pts,))
        [(x1, y1), (x2, y2)] = pts 
        xm = min(x1,x2); xM = max(x1,x2)
        ym = min(y1,y2); yM = max(y1,y2)
        xywh = map(int, map(round,(xm, ym, xM-xm, yM-ym)))
        roi = np.array(xywh, dtype=np.int32)
        print('[*guitools] roi = %r ' % (roi,))
        return roi
    except Exception as ex:
        print('[*guitools] ROI selection Failed:\n%r' % (ex,))
        return None

def _addOptions(msgBox, options):
    #msgBox.addButton(Qt.QMessageBox.Close)
    for opt in options:
        role = QtGui.QMessageBox.ApplyRole
        msgBox.addButton(QtGui.QPushButton(opt), role)

def _cacheReply(msgBox):
    dontPrompt = QtGui.QCheckBox('dont ask me again', parent=msgBox)
    dontPrompt.blockSignals(True)
    msgBox.addButton(dontPrompt, Qt.QMessageBox.ActionRole)
    return dontPrompt

def _newMsgBox(msg='', title='', parent=None, options=None, cache_reply=False):
    msgBox = QtGui.QMessageBox(parent)
    #msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    #std_buts = Qt.QMessageBox.Close
    #std_buts = Qt.QMessageBox.NoButton
    std_buts = Qt.QMessageBox.Cancel
    msgBox.setStandardButtons(std_buts)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(parent is not None)
    return msgBox

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

def user_input(parent, msg, title='input dialog'):
    reply, ok = QtGui.QInputDialog.getText(parent, title, msg)
    if not ok: return None
    return str(reply)

def user_info(parent, msg, title='info'):
    msgBox = _newMsgBox(msg, title, parent)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(Qt.QMessageBox.Ok)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()

def user_option(parent, msg, title='options', options=['No', 'Yes'], use_cache=False):
    'Prompts user with several options with ability to save decision'
    print('[*guitools] user_option:\n %r: %s'+title+': '+msg)
    # Recall decision
    cache_id = helpers.hashstr(title+msg)
    if use_cache:
        reply = io.global_cache_read(cache_id, default=None)
        if reply is not None: return reply
    # Create message box
    msgBox = _newMsgBox(msg, title, parent)
    _addOptions(msgBox, options)
    if use_cache:
        dontPrompt = _cacheReply(msgBox)
    # Wait for output
    optx = msgBox.exec_()
    if optx == Qt.QMessageBox.Cancel:
        return None
    try:
        reply = options[optx]
    except Exception as ex:
        print('[*guitools] USER OPTION EXCEPTION !')
        print('[*guitools] optx = %r' % optx)
        print('[*guitools] options = %r' % options)
        print('[*guitools] ex = %r' % ex)
        raise
    # Remember decision
    if use_cache and dontPrompt.isChecked():
        io.global_cache_write(cache_id, reply)
    del msgBox
    return reply

def user_question(msg):
    msgBox = Qt.QMessageBox.question(None, '', 'lovely day?')

def getQtImageNameFilter():
    imgNamePat = ' '.join(['*'+ext for ext in helpers.IMG_EXTENSIONS])
    imgNameFilter = 'Images (%s)' % (imgNamePat)
    return imgNameFilter

def select_images(caption='Select images:', directory=None):
    name_filter = getQtImageNameFilter()
    return select_files(caption, directory, name_filter)

def select_files(caption='Select Files:', directory=None, name_filter=None):
    'Selects one or more files from disk using a qt dialog'
    print(caption)
    if directory is None: 
        directory = io.global_cache_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory, filter=name_filter)
    file_list = map(str, qfile_list)
    print('Selected %d files' % len(file_list))
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
    print('[*guitools] cached db_dir=%r' % db_dir)
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
        print('[*guitools] Initializing QApplication')
        app = Qt.QApplication(sys.argv)
    try:
        __IPYTHON__
        is_root = False
    # You are not root if you are in IPYTHON
    except NameError as ex:
        pass
    IS_INIT = True
    return app, is_root


def exit_application():
    print('[*guitools] exiting application')
    QtGui.qApp.quit()

def run_main_loop(app, is_root=True, backend=None):
    if backend is not None:
        print('[*guitools] setting active window')
        app.setActiveWindow(backend.win)
    if is_root:
        print('[*guitools] running main loop.')
        timer = ping_python_interpreter()
        sys.exit(app.exec_())
    else:
        print('[*guitools] using roots main loop')

def ping_python_interpreter(frequency=100):
    'Create a QTimer which lets the python intepreter run every so often'
    timer = Qt.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer

def make_dummy_main_window():
    class DummyBackend(Qt.QObject):
        def __init__(self):
            super(DummyBackend,  self).__init__()
            self.win = PyQt4.Qt.QMainWindow()
            self.win.setWindowTitle('Dummy Main Window')
            self.win.show()
    backend = DummyBackend()
    return backend

def make_main_window(hs=None):
    import guiback
    backend = guiback.MainWindowBackend(hs=hs)
    backend.win.show()
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
    app, is_root = init_qtapp()
    backend = make_dummy_main_window()
    win = backend.win
    run_main_loop(app, is_root, backend)
