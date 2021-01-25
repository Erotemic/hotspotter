
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[guitools]')
# Python
from os.path import split
import sys
#import warnings
# Science
import numpy as np
# Qt
if 0:
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtCore import Qt
    from PyQt5.QtGui import QApplication
    QtWidgets = QtGui
else:
    from matplotlib.backends import backend_qt5 as backend_qt
    from PyQt5 import QtCore
    from PyQt5 import QtGui
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtWidgets import QApplication
    from PyQt5 import QtWidgets

# HotSpotter
from hscom import fileio as io
from hscom import helpers
from hscom import helpers as util
from hsviz import draw_func2 as df2

IS_INIT = False
QAPP = None
IS_ROOT = False
DISABLE_NODRAW = False
DEBUG = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


def configure_matplotlib():
    import multiprocessing
    import matplotlib
    mplbackend = matplotlib.get_backend()
    if multiprocessing.current_process().name == 'MainProcess':
        print('[*guitools] current mplbackend is: %r' % mplbackend)
        print('[*guitools] matplotlib.use(Qt4Agg)')
    else:
        return
    matplotlib.rcParams['toolbar'] = 'toolbar2'
    matplotlib.rc('text', usetex=False)
    #matplotlib.rcParams['text'].usetex = False
    if mplbackend != 'Qt4Agg':
        matplotlib.use('Qt4Agg', warn=True, force=True)
        mplbackend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            print('[*guitools] current mplbackend is: %r' % mplbackend)
        #matplotlib.rcParams['toolbar'] = 'None'
        #matplotlib.rcParams['interactive'] = True


#---------------
# SLOT DECORATORS


def slot_(*types, **kwargs_):  # This is called at wrap time to get args
    '''
    wrapper around pyqtslot decorator
    *args = types
    kwargs_['initdbg']
    kwargs_['rundbg']
    '''
    initdbg = kwargs_.get('initdbg', DEBUG)
    rundbg  = kwargs_.get('rundbg', DEBUG)

    # Wrap with debug statments
    def pyqtSlotWrapper(func):
        func_name = func.__name__
        if initdbg:
            print('[@guitools] Wrapping %r with slot_' % func.__name__)

        if rundbg:
            @QtCore.pyqtSlot(*types, name=func.__name__)
            def slot_wrapper(self, *args, **kwargs):
                argstr_list = list(map(str, args))
                kwastr_list = ['%s=%s' % item for item in kwargs.items()]
                argstr = ', '.join(argstr_list + kwastr_list)
                print('[**slot_.Begining] %s(%s)' % (func_name, argstr))
                #with helpers.Indenter():
                result = func(self, *args, **kwargs)
                print('[**slot_.Finished] %s(%s)' % (func_name, argstr))
                return result
        else:
            @QtCore.pyqtSlot(*types, name=func.__name__)
            def slot_wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                return result

        slot_wrapper.__name__ = func_name
        return slot_wrapper
    return pyqtSlotWrapper


#/SLOT DECORATOR
#---------------


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblocking(func):
    #printDBG('[@guitools] Wrapping %r with backblocking' % func.func_name)

    def block_wrapper(back, *args, **kwargs):
        #print('[guitools] BLOCKING')
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            back.front.blockSignals(wasBlocked_)
            print('Block wrapper caugt exception in %r' % func.__name__)
            print('back = %r' % back)
            VERBOSE = False
            if VERBOSE:
                print('*args = %r' % (args,))
                print('**kwargs = %r' % (kwargs,))
            #print('ex = %r' % ex)
            import traceback
            print(traceback.format_exc())
            #back.user_info('Error in blocking ex=%r' % ex)
            back.user_info('Error while blocking gui:\nex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        #print('[guitools] UNBLOCKING')
        return result
    block_wrapper.__name__ = func.__name__
    return block_wrapper


def frontblocking(func):
    # HACK: blocking2 is specific to fron
    #printDBG('[@guitools] Wrapping %r with frontblocking' % func.func_name)

    def block_wrapper(front, *args, **kwargs):
        #print('[guitools] BLOCKING')
        #wasBlocked = self.blockSignals(True)
        wasBlocked_ = front.blockSignals(True)
        try:
            result = func(front, *args, **kwargs)
        except Exception as ex:
            front.blockSignals(wasBlocked_)
            print('Block wrapper caught exception in %r' % func.__name__)
            print('front = %r' % front)
            VERBOSE = False
            if VERBOSE:
                print('*args = %r' % (args,))
                print('**kwargs = %r' % (kwargs,))
            #print('ex = %r' % ex)
            front.user_info('Error in blocking ex=%r' % ex)
            raise
        front.blockSignals(wasBlocked_)
        #print('[guitools] UNBLOCKING')
        return result
    block_wrapper.__name__ = func.__name__
    return block_wrapper


# DRAWING DECORATOR
def drawing(func):
    'Wraps a class function and draws windows on completion'
    #printDBG('[@guitools] Wrapping %r with drawing' % func.func_name)
    @util.indent_decor('[drawing]')
    def drawing_wrapper(self, *args, **kwargs):
        #print('[guitools] DRAWING')
        result = func(self, *args, **kwargs)
        #print('[guitools] DONE DRAWING')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()
        return result
    drawing_wrapper.__name__ = func.__name__
    return drawing_wrapper


@profile
def select_orientation():
    #from matplotlib.backend_bases import mplDeprecation
    print('[*guitools] Define an orientation angle by clicking two points')
    try:
        # Compute an angle from user interaction
        sys.stdout.flush()
        fig = df2.gcf()
        oldcbid, oldcbfn = df2.disconnect_callback(fig, 'button_press_event')
        #with warnings.catch_warnings():
            #warnings.filterwarnings("ignore", category=mplDeprecation)
        pts = np.array(fig.ginput(2))
        #print('[*guitools] ginput(2) = %r' % pts)
        # Get reference point to origin
        refpt = pts[1] - pts[0]
        #theta = np.math.atan2(refpt[1], refpt[0])
        theta = np.math.atan2(refpt[1], refpt[0])
        print('The angle in radians is: %r' % theta)
        df2.connect_callback(fig, 'button_press_event', oldcbfn)
        return theta
    except Exception as ex:
        print('Annotate Orientation Failed %r' % ex)
        return None


@profile
def select_roi():
    #from matplotlib.backend_bases import mplDeprecation
    print('[*guitools] Define a Rectanglular ROI by clicking two points.')
    try:
        sys.stdout.flush()
        fig = df2.gcf()
        # Disconnect any other button_press events
        oldcbid, oldcbfn = df2.disconnect_callback(fig, 'button_press_event')
        #with warnings.catch_warnings():
            #warnings.filterwarnings("ignore", category=mplDeprecation)
        pts = fig.ginput(2)
        print('[*guitools] ginput(2) = %r' % (pts,))
        [(x1, y1), (x2, y2)] = pts
        xm = min(x1, x2)
        xM = max(x1, x2)
        ym = min(y1, y2)
        yM = max(y1, y2)
        xywh = list(map(int, list(map(round, (xm, ym, xM - xm, yM - ym)))))
        roi = np.array(xywh, dtype=np.int32)
        # Reconnect the old button press events
        df2.connect_callback(fig, 'button_press_event', oldcbfn)
        print('[*guitools] roi = %r ' % (roi,))
        return roi
    except Exception as ex:
        print('[*guitools] ROI selection Failed:\n%r' % (ex,))
        return None


def _addOptions(msgBox, options):
    #msgBox.addButton(QtWidgets.QMessageBox.Close)
    for opt in options:
        role = QtWidgets.QMessageBox.ApplyRole
        msgBox.addButton(QtWidgets.QPushButton(opt), role)


def _cacheReply(msgBox):
    dontPrompt = QtWidgets.QCheckBox('dont ask me again', parent=msgBox)
    dontPrompt.blockSignals(True)
    msgBox.addButton(dontPrompt, QtWidgets.QMessageBox.ActionRole)
    return dontPrompt


def _newMsgBox(msg='', title='', parent=None, options=None, cache_reply=False):
    msgBox = QtWidgets.QMessageBox(parent)
    #msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    #std_buts = QtWidgets.QMessageBox.Close
    #std_buts = QtWidgets.QMessageBox.NoButton
    std_buts = QtWidgets.QMessageBox.Cancel
    msgBox.setStandardButtons(std_buts)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(parent is not None)
    return msgBox


@profile
def msgbox(msg, title='msgbox'):
    'Make a non modal critical QtWidgets.QMessageBox.'
    msgBox = QtWidgets.QMessageBox(None)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox


def user_input(parent, msg, title='input dialog'):
    reply, ok = QtWidgets.QInputDialog.getText(parent, title, msg)
    if not ok:
        return None
    return str(reply)


def user_info(parent, msg, title='info'):
    msgBox = _newMsgBox(msg, title, parent)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()


@profile
def _user_option(parent, msg, title='options', options=['No', 'Yes'], use_cache=False):
    'Prompts user with several options with ability to save decision'
    print('[*guitools] _user_option:\n %r: %s' + title + ': ' + msg)
    # Recall decision
    print('[*guitools] asking user: %r %r' % (msg, title))
    cache_id = helpers.hashstr(title + msg)
    if use_cache:
        reply = io.global_cache_read(cache_id, default=None)
        if reply is not None:
            return reply
    # Create message box
    msgBox = _newMsgBox(msg, title, parent)
    _addOptions(msgBox, options)
    if use_cache:
        dontPrompt = _cacheReply(msgBox)
    # Wait for output
    optx = msgBox.exec_()
    if optx == QtWidgets.QMessageBox.Cancel:
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
    msgBox = QtWidgets.QMessageBox.question(None, '', 'lovely day?')
    return msgBox


def getQtImageNameFilter():
    imgNamePat = ' '.join(['*' + ext for ext in helpers.IMG_EXTENSIONS])
    imgNameFilter = 'Images (%s)' % (imgNamePat)
    return imgNameFilter


@profile
def select_images(caption='Select images:', directory=None):
    name_filter = getQtImageNameFilter()
    selected = select_files(caption, directory, name_filter)
    print('selected = {!r}'.format(selected))
    return selected


@profile
def select_files(caption='Select Files:', directory=None, name_filter=None):
    'Selects one or more files from disk using a qt dialog'
    print(caption)
    if directory is None:
        directory = io.global_cache_read('select_directory')
    qdlg = QtWidgets.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory, filter=name_filter)
    print('qfile_list = {!r}'.format(qfile_list))
    file_list = list(map(str, qfile_list))
    print('Selected %d files' % len(file_list))
    io.global_cache_write('select_directory', directory)
    return file_list


@profile
def select_directory(caption='Select Directory', directory=None):
    print(caption)
    if directory is None:
        directory = io.global_cache_read('select_directory')
    qdlg = QtWidgets.QFileDialog()
    qopt = QtWidgets.QFileDialog.ShowDirsOnly
    qdlg_kwargs = dict(caption=caption, options=qopt, directory=directory)
    dpath = str(qdlg.getExistingDirectory(**qdlg_kwargs))
    print('Selected Directory: %r' % dpath)
    io.global_cache_write('select_directory', split(dpath)[0])
    return dpath


@profile
def show_open_db_dlg(parent=None):
    # OLD
    from ._frontend import OpenDatabaseDialog
    if not '-nc' in sys.argv and not '--nocache' in sys.argv:
        db_dir = io.global_cache_read('db_dir')
        if db_dir == '.':
            db_dir = None
    print('[*guitools] cached db_dir=%r' % db_dir)
    if parent is None:
        parent = QtWidgets.QDialog()
    opendb_ui = OpenDatabaseDialog.Ui_Dialog()
    opendb_ui.setupUi(parent)
    #opendb_ui.new_db_but.clicked.connect(create_new_database)
    #opendb_ui.open_db_but.clicked.connect(open_old_database)
    parent.show()
    return opendb_ui, parent


@util.indent_decor('[qt-init]')
@profile
def init_qtapp():
    global IS_INIT
    global IS_ROOT
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT
    app = QtCore.QCoreApplication.instance()
    is_root = app is None
    if is_root:  # if not in qtconsole
        print('[*guitools] Initializing QApplication')
        app = QApplication(sys.argv)
        QAPP = app
    try:
        __IPYTHON__
        is_root = False
    # You are not root if you are in IPYTHON
    except NameError:
        pass
    IS_INIT = True
    return app, is_root


@util.indent_decor('[qt-exit]')
@profile
def exit_application():
    print('[*guitools] exiting application')
    QtWidgets.qApp.quit()


@util.indent_decor('[qt-main]')
@profile
def run_main_loop(app, is_root=True, back=None, **kwargs):
    if back is not None:
        print('[*guitools] setting active window')
        app.setActiveWindow(back.front)
        back.timer = ping_python_interpreter(**kwargs)
    if is_root:
        exec_core_app_loop(app)
        #exec_core_event_loop(app)
    else:
        print('[*guitools] using roots main loop')


@profile
def exec_core_event_loop(app):
    # This works but does not allow IPython injection
    print('[*guitools] running core application loop.')
    try:
        from IPython.lib.inputhook import enable_qt4
        enable_qt4()
        from IPython.lib.guisupport import start_event_loop_qt4
        print('Starting ipython qt4 hook')
        start_event_loop_qt4(app)
    except ImportError:
        pass
    app.exec_()


@profile
def exec_core_app_loop(app):
    # This works but does not allow IPython injection
    print('[*guitools] running core application loop.')
    app.exec_()
    #sys.exit(app.exec_())


@profile
def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer


def make_dummy_main_window():
    class DummyBackend(QtCore.QObject):
        def __init__(self):
            super(DummyBackend,  self).__init__()
            self.front = QtWidgets.QMainWindow()
            self.front.setWindowTitle('Dummy Main Window')
            self.front.show()
    back = DummyBackend()
    return back


def get_scope(qobj, scope_title='_scope_list'):
    if not hasattr(qobj, scope_title):
        setattr(qobj, scope_title, [])
    return getattr(qobj, scope_title)


def clear_scope(qobj, scope_title='_scope_list'):
    setattr(qobj, scope_title, [])


def enfore_scope(qobj, scoped_obj, scope_title='_scope_list'):
    get_scope(qobj, scope_title).append(scoped_obj)


@profile
def popup_menu(widget, opt2_callback, parent=None):
    def popup_slot(pos):
        print(pos)
        menu = QtWidgets.QMenu()
        actions = [menu.addAction(opt, func) for opt, func in
                   iter(opt2_callback)]
        #pos=QtWidgets.QCursor.pos()
        selection = menu.exec_(widget.mapToGlobal(pos))
        return selection, actions
    if parent is not None:
        # Make sure popup_slot does not lose scope.
        for _slot in get_scope(parent, '_popup_scope'):
            parent.customContextMenuRequested.disconnect(_slot)
        clear_scope(parent, '_popup_scope')
        parent.setContextMenuPolicy(Qt.CustomContextMenu)
        parent.customContextMenuRequested.connect(popup_slot)
        enfore_scope(parent, popup_slot, '_popup_scope')
    return popup_slot


@profile
def make_header_lists(tbl_headers, editable_list, prop_keys=[]):
    col_headers = tbl_headers[:] + prop_keys
    col_editable = [False] * len(tbl_headers) + [True] * len(prop_keys)
    for header in editable_list:
        col_editable[col_headers.index(header)] = True
    return col_headers, col_editable
