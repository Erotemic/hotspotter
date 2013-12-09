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
    gui_back = MainWindowBackend(hs=hs)
    #main_win.draw_splash()
    gui_back.win.show()
    return gui_back

class MainWindowBackend(QtCore.QObject):
    ''' Class used by the backend to send and recieve signals to and from the
    frontend'''
    populateSignal = pyqtSignal(str, list, list, list, list)
    def __init__(self, hs=None):
        super(MainWindowBackend, self).__init__()
        print('[back] creating backend')
        self.hs = hs
        self.win = MainWindowFrontend(gui_back=self)
        df2.register_matplotlib_widget(self.win.plotWidget)
        # connect signals
        self.populateSignal.connect(self.win.populateSlot)
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
        cx2_cid  = self.hs.tables.cx2_cid
        cx2_nx   = self.hs.tables.cx2_nx
        cx2_gx   = self.hs.tables.cx2_gx
        nx2_name = self.hs.tables.nx2_name
        gx2_gname = self.hs.tables.gx2_gname
        cx_list = [cx for cx, cid in enumerate(cx2_cid) if cid > 0]
        xs_list = [(cx, cx2_nx[cx], cx2_gx[cx],) for cx in iter(cx_list)]
        row2_datatup = [(cx2_cid[cx], nx2_name[nx], gx2_gname[gx],) for (cx, nx, gx) in iter(xs_list)]
        row_list  = range(len(row2_datatup))
        self.populateSignal.emit('chip', col_headers, col_editable, row_list, row2_datatup)

    @pyqtSlot(name='open_database')
    def open_database(self):
        print('[back] open_database')
        import HotSpotter
        try:
            args = self.hs.args # Take previous args
            # Ask user for db
            db_dir = select_directory('Select (or create) a database directory.')
            print('[main] user selects database: '+db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, db_dir=db_dir)
            hs.load()
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            self.connect_api(hs)
        except Exception as ex:
            print('aborting open database')
            print(repr(ex))
            if hs.args.strict: raise

    @pyqtSlot(name='save_database')
    def save_database(self):
        print('[back] save_database')
        raise NotImplemented('');

    @pyqtSlot(name='quit')
    def quit(self):
         QtGui.qApp.quit

    def add_images_from_dir(self):
        img_dpath = select_directory('Select directory with images in it')

    def draw_splash(self):
        print('[back] draw_splash()')
        img = imread('_frontend/splash.png')
        fig = df2.figure()
        print(fig)
        print(fig is self.win.plotWidget.figure)
        df2.imshow(img)
        df2.update()
    
    def show(self):
        self.win.show()

    @pyqtSlot(str, name='backend_print')
    def backend_print(msg):
        print(msg)
