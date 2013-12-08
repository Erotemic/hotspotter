from __future__ import division, print_function
import __builtin__
import fileio as io
import os
import os.path
import sys
from HotSpotter import imread
import draw_func2 as df2
import PyQt4
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.Qt import (QMainWindow, QApplication, QCoreApplication,
                      QTableWidgetItem, QAbstractItemView, QWidget, Qt,
                      pyqtSlot, pyqtSignal, QStandardItem, QStandardItemModel,
                      QString, QObject, QInputDialog, QDialog, QTreeWidgetItem)
from _tpl.other.matplotlibwidget import MatplotlibWidget

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[gui] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

IS_INIT = False

def make_main_window(hs=None):
    main_win = MainWindowBackend(hs)
    #main_win.draw_splash()
    main_win.show()
    return main_win

class MainWindowBackend(QtCore.QObject):
    ''' Class used by the backend to send and recieve signals to and from the
    frontend'''
    populateSignal = pyqtSignal(str, list, list, list, list)
    def __init__(self, hs=None):
        super(MainWindowBackend, self).__init__()
        print('[back] creating backend')
        self.hs = hs
        self.win = MainWindowFrontend()
        df2.register_matplotlib_widget(self.win.plotWidget)
        # connect signals
        self.populateSignal.connect(self.win.populate_table_slot)
        if hs is not None:
            self.connect_api(hs)
            
    def connect_api(self, hs):
        print('[win] connecting api')
        self.hs = hs
        self.populate_image_table()
        self.populate_chip_table()
        #self.database_loaded.emit()

    def populate_image_table(self):
        print('[win] populate_image_table()')
        #col_headers  = ['Image ID', 'Image Name', 'Chip IDs', 'Chip Names']
        #col_editable = [ False    ,  False      ,  False    ,  False      ]
        col_headers   = ['Image Name']
        col_editable  = [False]
        # Populate table with valid image indexes
        gx2_gname = self.hs.tables.gx2_gname
        row_list  = range(len(gx2_gname))
        row2_datatup = [(gname,) for gname in gx2_gname]
        self.populateSignal.emit('image', col_headers, col_editable, row_list, row2_datatup)

    def populate_chip_table(self):
        print('[win] populate_chip_table()')
        col_headers  = ['Chip ID', 'Name', 'Image']
        col_editable = [ False  ,    True,   False]
        # Add User Properties to headers
        #col_headers  += hs.tables.cx2_px.user_props.keys()
        #col_editable += [True for key in cm.user_props.keys()]
        # Populate table with valid image indexes
        cx2_cid = self.hs.tables.cx2_cid
        cx2_nx = self.hs.tables.cx2_nx
        cx2_gx = self.hs.tables.cx2_gx
        nx2_name = self.hs.tables.nx2_name
        gx2_gname = self.hs.tables.gx2_gname
        cx_list = [cx for cx, cid in enumerate(cx2_cid) if cid > 0]
        xs_list = [(cx, cx2_nx[cx], cx2_gx[cx],) for cx in iter(cx_list)]
        row2_datatup = [(cx2_cid[cx], nx2_name[nx], gx2_gname[gx],) for (cx, nx, gx) in iter(xs_list)]
        row_list  = range(len(row2_datatup))
        self.populateSignal.emit('chip', col_headers, col_editable, row_list, row2_datatup)

    def open_database(self):
        import HotSpotter
        try:
            args = hs.args # Take previous args
            # Ask user for db
            db_dir = gui.select_directory('Select (or create) a database directory.')
            print('[main] user selects database: '+db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, dbdir=dbdir)
            hs.load()
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            self.connect_api(hs)
        except Exception as ex:
            print(ex)

    def add_images_from_dir(self):
        img_dpath = select_directory('Select directory with images in it')

    def draw_splash(self):
        img = imread('_frontend/splash.png')
        fig = df2.figure()
        print(fig)
        print(fig is self.win.plotWidget.figure)
        df2.imshow(img)
        df2.update()
    
    def show(self):
        self.win.show()

class MainWindowFrontend(QtGui.QMainWindow):
    populateChipTblSignal   = pyqtSignal(list, list, list, list)
    printSignal = pyqtSignal(str)
    def __init__(self):
        from _frontend.MainSkel import Ui_mainSkel
        super(MainWindowFrontend, self).__init__()
        print('[front] creating frontend')
        self.ui=Ui_mainSkel()
        self.ui.setupUi(self)
        self.plotWidget = MatplotlibWidget(self.ui.centralwidget)
        self.plotWidget.setObjectName(_fromUtf8('plotWidget'))
        self.ui.root_hlayout.addWidget(self.plotWidget)
        # connect signals
        self.printSignal.connect(backend_print)
        #def ui_connect(action, event):
            #self.ui.__dict__[action].triggered.connect(event)
        #ui_connect('actionQuit', QtGui.qApp.quit)
        self.ui.actionQuit.triggered.connect(QtGui.qApp.quit)

    def print(self, msg):
        self.printSignal.emit('[front] '+msg)

    def closeEvent(self, event):
        print('Close Event')
        self.print('Close Event')
        # Close the app so python does not crash
        QtGui.qApp.quit()
        event.ignore()

    @pyqtSlot(str, list, list, list, list, name='populate_table_slot')
    def populateSlot(self, table_name, col_headers, col_editable, row_list, row2_datatup):
        print('populateSlot()')
        self.print('populateSlot()')
        table_name = str(table_name)
        try:
            tbl = self.ui.__dict__[table_name+'_TBL']
        except KeyError as ex:
            valid_table_names = [key for key in self.ui.__dict__.keys() 
                                 if key.find('_TBL') >= 0]
            msg = '\n'.join(['Invalid table_name = '+table_name+'_TBL',
                             'valid names:\n  '+'\n  '.join(valid_table_names)])
            raise Exception(msg)
        self._populate_table(tbl, col_headers, col_editable, row_list, row2_datatup)

    def _populate_table(self, tbl, col_headers, col_editable, row_list, row2_datatup):
        print('_populate_table()')
        self.print('_populate_table()')
        #tbl = main_skel.chip_TBL
        hheader = tbl.horizontalHeader()
        sort_col = hheader.sortIndicatorSection()
        sort_ord = hheader.sortIndicatorOrder()
        tbl.sortByColumn(0, Qt.AscendingOrder) # Basic Sorting
        prevBlockSignals = tbl.blockSignals(True)
        tbl.clear()
        tbl.setColumnCount(len(col_headers))
        tbl.setRowCount(len(row_list))
        tbl.verticalHeader().hide()
        tbl.setHorizontalHeaderLabels(col_headers)
        tbl.setSelectionMode( QAbstractItemView.SingleSelection )
        tbl.setSelectionBehavior( QAbstractItemView.SelectRows)
        tbl.setSortingEnabled(False)
        for row in iter(row_list):
            data_tup = row2_datatup[row]
            for col, data in enumerate(data_tup):
                item = QTableWidgetItem()
                try:
                    int_data = int(data)
                    item.setData(Qt.DisplayRole, int_data)
                except ValueError: # for strings
                    item.setText(str(data))
                except TypeError: #for lists
                    item.setText(str(data))
                item.setTextAlignment(Qt.AlignHCenter)
                if col_editable[col]: item.setFlags(item.flags() | Qt.ItemIsEditable)
                else: item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                tbl.setItem(row, col, item)
        tbl.setSortingEnabled(True)
        tbl.sortByColumn(sort_col,sort_ord) # Move back to old sorting
        tbl.show()
        tbl.blockSignals(prevBlockSignals)



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
    dpath = str(qdlg.getExistingDirectory(caption=caption, options=qopt, directory=directory))
    print('Selected Directory: '+dpath)
    io.global_cache_write('select_directory', os.path.split(dpath)[0])
    return dpath

@pyqtSlot(str, name='backend_print')
def backend_print(msg):
    print(msg)

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
        print('[gui] Initializing QApplication')
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

def run_main_loop(app, is_root=True, main_win=None):
    if main_win is not None:
        print('[gui] setting active window')
        app.setActiveWindow(main_win.win)
    if is_root:
        print('[gui] running main loop.')
        sys.exit(app.exec_())
    else:
        print('[gui] using roots main loop')

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

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    def test():
        app, is_root = init_qtapp()
        main_win = make_main_window()
        run_main_loop(app, is_root, main_win)
    test()
