from __future__ import division, print_function
from itertools import izip
from os.path import split, exists, join
# Qt
from PyQt4 import QtCore
from PyQt4.Qt import pyqtSlot, pyqtSignal
import numpy as np
# Hotspotter
import guifront
import guitools
import helpers
import fileio as io
import draw_func2 as df2
import vizualizations as viz
import HotSpotter


def rrr():
    'Dynamic module reloading'
    import imp
    import sys
    print('[*back] reloading ' + __name__)
    imp.reload(sys.modules[__name__])

DISABLE_NODRAW = True


class MainWindowBackend(QtCore.QObject):
    'Sends and recieves signals to and from the frontend'
    #--------------------------------------------------------------------------
    # Backend Signals
    populateSignal = pyqtSignal(str, list, list, list, list)
    setEnabledSignal = pyqtSignal(bool)
    setPlotWidgetEnabledSignal = pyqtSignal(bool)

    #--------------------------------------------------------------------------
    # Constructor
    def __init__(self, hs=None, app=None):
        super(MainWindowBackend, self).__init__()
        print('[*back] creating backend')
        self.hs  = hs
        self.app = app
        self.timer = None
        kwargs_ = {'use_plot_widget': False}
        self.win = guifront.MainWindowFrontend(backend=self, **kwargs_)
        self.selection = None
        if kwargs_['use_plot_widget']:
            df2.register_matplotlib_widget(self.win.plotWidget)
        df2.register_qt4_win(self.win)
        # connect signals
        self.populateSignal.connect(self.win.populate_tbl)
        self.setEnabledSignal.connect(self.win.setEnabled)
        self.setPlotWidgetEnabledSignal.connect(self.win.setPlotWidgetEnabled)
        if hs is not None:
            self.connect_api(hs)

    #------------------------
    # Draw Functions
    #------------------------
    def show_splash(self, fnum=1, view='Nice', **kwargs):
        fig = df2.figure(fignum=fnum, doclf=True)
        fig.clf()
        viz.show_splash(fnum=fnum)
        df2.set_figtitle('%s View' % view)
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()

    def show_image(self, gx, sel_cxs=[], **kwargs):
        fig = df2.figure(fignum=1, doclf=True)
        fig.clf()
        cx_clicked_func = lambda cx: self.select_gx(gx, cx)
        viz.show_image(self.hs, gx, sel_cxs, cx_clicked_func,
                       fnum=1, figtitle='Image View')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()

    def show_chip(self, cx, **kwargs):
        fig = df2.figure(fignum=2, doclf=True)
        fig.clf()
        INTERACTIVE_CHIPS = True  # This should always be True
        if INTERACTIVE_CHIPS:
            interact_fn = viz.show_chip_interaction
            interact_fn(self.hs, cx, fnum=2, figtitle='Chip View')
        else:
            viz.show_chip(self.hs, cx, fnum=2, figtitle='Chip View')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()

    def show_query(self, res, **kwargs):
        print('[back*] show_query()')
        fig = df2.figure(fignum=3, doclf=True)
        fig.clf()
        res.show_top(self.hs, fignum=3, figtitle='Query View ')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()

    #----------------------
    # Work Functions
    #----------------------
    def get_selected_gx(self):
        'selected image index'
        if self.selection is None:
            return None
        type_ = self.selection['type_']
        if type_ == 'gx':
            gx = self.selection['index']
        if type_ == 'cx':
            cx = self.selection['index']
            gx = self.hs.tables.cx2_gx(cx)
        return gx

    def get_selected_cx(self):
        'selected chip index'
        if self.selection is None:
            return None
        type_ = self.selection['type_']
        if type_ == 'cx':
            cx = self.selection['index']
        if type_ == 'gx':
            cx = self.selection['sub']
        return cx

    def update_window_title(self):
        if self.hs is None:
            title = 'Hotspotter - NULL database'
        if self.hs.dirs is None:
            title = 'Hotspotter - invalid database'
        else:
            db_dir = self.hs.dirs.db_dir
            db_name = split(db_dir)[1]
            title = 'Hotspotter - %r - %s' % (db_name, db_dir)
        self.win.setWindowTitle(title)

    def connect_api(self, hs):
        print('[*back] connect_api()')
        self.hs = hs
        if hs.tables is not None:
            self.populate_image_table()
            self.populate_chip_table()
            self.setEnabledSignal.emit(True)
            self.clear_selection()
            self.update_window_title()
        else:
            self.setEnabledSignal.emit(False)
        #self.database_loaded.emit()

    def populate_image_table(self):
        print('[*back] populate_image_table()')
        col_headers  = ['Image Index', 'Image Name', 'Num Chips']
        col_editable = [False,                False,       False]
        # Populate table with valid image indexes
        gx2_gname = self.hs.tables.gx2_gname
        gx2_cxs   = self.hs.gx2_cxs
        gx_list  = self.hs.get_valid_gxs()
        gname_list = [gx2_gname[gx] for gx in iter(gx_list)]
        num_cxs_list = [len(gx2_cxs(gx)) for gx in iter(gx_list)]
        row2_datatup = [tup for tup in izip(gx_list, gname_list, num_cxs_list)]
        row_list  = range(len(row2_datatup))
        self.populateSignal.emit('image', col_headers, col_editable,
                                 row_list, row2_datatup)

    def populate_chip_table(self):
        print('[*back] populate_chip_table()')
        col_headers  = ['Chip ID', 'Name', 'Image', 'Num Indexed Others']
        col_editable = [False,       True,   False,                False]
        # Add User Properties to headers
        prop_dict = self.hs.tables.prop_dict
        prop_keys = prop_dict.keys()
        col_headers  += prop_keys
        col_editable += [True] * len(prop_keys)
        # Populate table with valid image indexes
        cx2_cid  = self.hs.tables.cx2_cid
        cx2_nx   = self.hs.tables.cx2_nx
        cx2_gx   = self.hs.tables.cx2_gx
        nx2_name = self.hs.tables.nx2_name
        gx2_gname = self.hs.tables.gx2_gname
        cx_list = self.hs.get_valid_cxs()
        # Build lists of column values
        cid_list   = [cx2_cid[cx] for cx in iter(cx_list)]
        name_list  = [nx2_name[cx2_nx[cx]] for cx in iter(cx_list)]
        image_list = [gx2_gname[cx2_gx[cx]] for cx in iter(cx_list)]
        gtcxs_list = [self.hs.get_other_indexed_cxs(cx) for cx in iter(cx_list)]
        ngt_list = [len(gtcxs) for gtcxs in iter(gtcxs_list)]
        prop_lists = [[prop_dict[key][cx] for cx in iter(cx_list)] for key in iter(prop_keys)]
        # Define order of columns
        unziped_list = [cid_list, name_list, image_list, ngt_list] + prop_lists
        row2_datatup = [tup for tup in izip(*unziped_list)]
        row_list = range(len(row2_datatup))
        self.populateSignal.emit('chip', col_headers, col_editable,
                                 row_list, row2_datatup)

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def _add_images(self, fpath_list):
        print('[*back] _add_images()')
        num_new = self.hs.add_images(fpath_list)
        if num_new > 0:
            self.populate_image_table()

    def user_info(self, *args, **kwargs):
        return guitools.user_info(self.win, *args, **kwargs)

    def user_input(self, *args, **kwargs):
        return guitools.user_input(self.win, *args, **kwargs)

    def user_option(self, *args, **kwargs):
        return guitools.user_option(self.win, *args, **kwargs)

    def get_work_directory(self, use_cache=True):
        print('[*back] get_work_directory()')
        cache_id = 'work_directory_cache_id'
        if use_cache:
            work_dir = io.global_cache_read(cache_id, default='.')
            if work_dir is not '.' and exists(work_dir):
                return work_dir
        msg_dir = 'Work directory not currently set. Select a work directory'
        work_dir = guitools.select_directory(msg_dir)
        if not exists(work_dir):
            msg_try = 'Directory %r does not exist.' % work_dir
            opt_try = ['Try Again']
            try_again = self.user_option(msg_try, 'get work dir failed', opt_try, False)
            if try_again == 'Try Again':
                return self.get_work_dir(use_cache)
        io.global_cache_write(cache_id, work_dir)
        return work_dir

    @pyqtSlot(str, name='backend_print')
    def backend_print(self, msg):
        msg = str(msg)
        print(msg)

    def set_state(self, state):
        print('[*back] set state = %r' % state)
        pass

    @pyqtSlot(name='clear_selection')
    def clear_selection(self, **kwargs):
        print('[*back] clear_selection()')
        self.selection = None
        self.show_splash(1, 'Image', dodraw=False)
        self.show_splash(2, 'Chip', dodraw=False)
        self.show_splash(3, 'Results', **kwargs)

    # Table selection
    @pyqtSlot(int, name='select_gx')
    def select_gx(self, gx, cx=None, **kwargs):
        print('[*back] select_gx(%r, %r)' % (gx, cx))
        if cx is None:
            cxs = self.hs.gx2_cxs(gx)
            if len(cxs > 0):
                cx = cxs[0]
        if cx is None:
            self.show_splash(2, 'Chip', dodraw=False)
        else:
            #cid = self.hs.tables.cx2_cid[cx]
            self.show_chip(cx, dodraw=False)
        highlight_cxs = [] if cx is None else [cx]
        self.selection = {'type_': 'gx', 'index': gx, 'sub': cx}
        self.show_image(gx, highlight_cxs, **kwargs)

    # Table selection
    @pyqtSlot(int, name='select_cid')
    def select_cid(self, cid, **kwargs):
        print('[*back] select_cid(%r)' % cid)
        cx = self.hs.cid2_cx(cid)
        gx = self.hs.tables.cx2_gx[cx]
        self.select_gx(gx, cx=cx, **kwargs)

    #--------------------------------------------------------------------------
    # File menu slots
    #--------------------------------------------------------------------------
    # File -> New Database
    @pyqtSlot(name='new_database')
    def new_database(self):
        print('[*back] new_database()')
        new_db = self.user_input('Enter the new database name')
        msg_put = 'Where should I put %r?' % new_db
        opt_put = ['Choose Directory', 'My Work Dir']
        reply = self.user_option(msg_put, 'options', opt_put, True)
        if reply == opt_put[1]:
            put_dir = self.get_work_directory()
        elif reply == opt_put[0]:
            put_dir = guitools.select_directory('Select where to put the new database')
        elif reply is None:
            print('[*back] abort new database()')
            return None
        else:
            raise Exception('Unknown reply=%r' % reply)

        new_db_dir = join(put_dir, new_db)

        # Check the put directory exists and the new database does not exist
        msg_try = None
        if not exists(put_dir):
            msg_try = 'Directory %r does not exist.' % put_dir
        elif exists(new_db_dir):
            msg_try = 'New Database %r already exists.' % new_db_dir
        if msg_try is not None:
            opt_try = ['Try Again']
            title_try = 'New Database Failed'
            try_again = self.user_option(msg_try, title_try, opt_try, False)
            if try_again == 'Try Again':
                return self.new_database()
        print('[*back] valid new_db_dir = %r' % new_db_dir)
        helpers.ensurepath(new_db_dir)
        self.open_database(new_db_dir)

    # File -> Open Database
    @pyqtSlot(name='open_database')
    def open_database(self, db_dir=None):
        print('[*back] open_database')
        # Try to load db
        try:
            # Use the same args in a new (opened) database
            args = self.hs.args
            if db_dir is None:
                db_dir = guitools.select_directory('Select (or create) a database directory.')
            print('[main] user selects database: ' + db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, db_dir=db_dir)
            hs.load(load_all=False)
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            self.connect_api(hs)
        except Exception as ex:
            print('aborting open database')
            print(ex)
            if self.hs.args.strict:
                raise

    # File -> Save Database
    @pyqtSlot(name='save_database')
    def save_database(self):
        print('[*back] save_database')
        self.hs.save_database()

    # File -> Import Images
    @pyqtSlot(name='import_images')
    def import_images(self):
        print('[*back] import images')
        msg = 'Import specific files or whole directory?'
        title = 'Import Images'
        options = ['Files', 'Directory']
        reply = self.user_option(msg, title, options, False)
        if reply == 'Files':
            self.add_images_from_files()
        if reply == 'Directory':
            self.add_images_from_dir()
        #raise NotImplementedError('')

    # File -> Import Images From File
    @pyqtSlot(name='import_images_from_file')
    def add_images_from_files(self):
        print('[*back] add_images_from_files()')
        fpath_list = guitools.select_images('Select image files to import')
        self._add_images(fpath_list)

    # File -> Import Images From Directory
    @pyqtSlot(name='import_images_from_dir')
    def add_images_from_dir(self):
        print('[*back] add_images_from_dir()')
        img_dpath = guitools.select_directory('Select directory with images in it')
        print('[*back] selected ' + img_dpath)
        fpath_list = helpers.list_images(img_dpath, fullpath=True)
        self._add_images(fpath_list)

    # File -> Quit
    @pyqtSlot(name='quit')
    def quit(self):
        print('[*back] quit()')
        guitools.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------
    # Action -> New Chip Property
    @pyqtSlot(name='new_prop')
    def new_prop(self):
        print('[*back] new_prop()')
        new_prop = self.user_input('What is the new property name?')
        self.hs.add_property(new_prop)
        self.populate_chip_table()

    # Action -> Add ROI
    @pyqtSlot(name='add_chip')
    def add_chip(self):
        print('[*back] add_chip()')
        gx = self.get_selected_gx()
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        cx = self.hs.add_chip(gx, roi)
        self.populate_image_table()
        self.populate_chip_table()
        self.select_gx(gx, cx)

    # Action -> Query
    @pyqtSlot(name='query')
    def query(self, cid=None):
        print('[*back] query()')
        if cid is not None:
            self.select_cid(cid, dodraw=False)
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot query. No chip selected')
            return
        res = self.hs.query(cx)
        self.show_query(res)
        return res

    # Action -> Reselect ROI
    @pyqtSlot(name='reselect_roi')
    def reselect_roi(self, **kwargs):
        print('[*back] reselect_roi()')
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot reselect ROI. No chip selected')
            return
        gx = self.hs.tables.cx2_gx[cx]
        self.show_image(gx, [cx], **kwargs)
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        self.hs.change_roi(cx, roi)
        self.populate_image_table()
        self.populate_chip_table()
        self.select_gx(gx, cx, **kwargs)
        pass

    # Action -> Reselect ORI
    @pyqtSlot(name='reselect_ori')
    def reselect_ori(self, **kwargs):
        print('[*back] reselect_ori()')
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot reselect orientation. No chip selected')
            return
        gx = self.hs.tables.cx2_gx[cx]
        self.show_image(gx, [cx], **kwargs)
        theta = guitools.select_ori()
        if theta is None:
            print('[back*] roiselection failed. Not adding')
            return
        self.hs.change_theta(cx, theta)
        self.populate_image_table()
        self.populate_chip_table()
        self.select_gx(gx, cx, **kwargs)
        pass

    # Change chip propery
    @pyqtSlot(int, str, str, name='change_chip_property')
    def change_chip_property(self, cid, key, val):
        key, val = map(str, (key, val))
        print('[*back] change_chip_property(%r, %r, %r)' % (cid, key, val))
        cx = self.hs.cid2_cx(cid)
        if key == 'Name':
            self.hs.change_name(cx, val)
        else:
            self.hs.change_prop(cx, key, val)
        self.populate_chip_table()

    @pyqtSlot(name='edit_preferences')
    def edit_preferences(self):
        print('[*back] edit_preferences')
        self.edit_prefs = self.hs.prefs.createQWidget()
        query_uid = ''.join(self.hs.prefs.query_cfg.get_uid())
        print('[*back] query_uid = %s' % query_uid)

    # Action -> Delete Chip
    @pyqtSlot(name='delete_chip')
    def delete_chip(self):
        print('[*back] delete_chip()')
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot delete chip. No chip selected')
            return
        gx = self.hs.cx2_gx(cx)
        self.hs.delete_chip(cx)
        self.populate_image_table()
        self.populate_chip_table()
        self.select_gx(gx)
        pass

    # Action -> Next
    @pyqtSlot(name='select_next')
    def select_next(self):
        print('[*back] select_next()')
        select_mode = 'in_order'  # 'unannotated'
        if select_mode == 'in_order':
            msg = self.select_next_in_order()
        elif select_mode == 'unannotated':
            msg = self.select_next_unannotated()
        else:
            raise Exception('uknown=%r' % select_mode)
        if msg is not None:
            self.user_info(msg)

    def select_next_unannotated(self):
        msg = 'err'
        if self.selection is None or self.selection['type_'] == 'gx':
            valid_gxs = self.hs.get_valid_gxs()
            has_chips = lambda gx: len(self.hs.gx2_cxs(gx)) > 0
            hascxs_list = map(has_chips, iter(valid_gxs))
            try:
                gx = valid_gxs[hascxs_list.index(False)]
                self.select_gx(gx)
                return
            except ValueError:
                msg = 'All images have detections. Excellent! '
        if self.selection is None or msg is not None and self.selection['type_'] == 'cx':
            valid_cxs = self.hs.get_valid_cxs()
            has_name = lambda cx: self.hs.cx2_name(cx) != '____'
            is_named = map(has_name, iter(valid_cxs))
            try:
                cx = valid_cxs[is_named.index(False)]
                cid = self.hs.tables.cx2_cid[cx]
                self.select_cid(cid)
                return
            except ValueError:
                msg = 'All chips are named. Awesome! '
        return msg

    def select_next_in_order(self):
        if self.selection is None:
            # No selection
            return self.select_next_unannotated()
        if self.selection['type_'] == 'gx':
            # Select next image
            gx = self.selection['index']
            gx2_gname = self.hs.tables.gx2_gname
            next_gx = gx + 1
            while next_gx < len(gx2_gname):
                if gx2_gname[next_gx] != '':
                    self.select_gx(next_gx)
                    break
                next_gx += 1
            return
        elif self.selection['type_'] == 'cx':
            # Select next chip
            cx = self.selection['index']
            cx2_cid = self.hs.tables.cx2_cid
            next_cx = cx + 1
            while next_cx < len(cx2_cid):
                cid = cx2_cid[next_cx]
                if cid != 0:
                    self.select_cid(cid)
                    break
                next_cx += 1
            return
        return 'end of the list'

    # Batch Actions
    @pyqtSlot(name='precompute_feats')
    def precompute_feats(self):
        print('[*back] precompute_feats()')
        prevBlock = self.win.blockSignals(True)
        self.hs.load_chips()
        self.hs.load_features()
        self.win.blockSignals(prevBlock)
        print('[*back] Finished precomputing features')

    @pyqtSlot(name='precompute_queries')
    def precompute_queries(self):
        #http://stackoverflow.com/questions/15637768/pyqt-how-to-capture-output-of-pythons-interpreter-and-display-it-in-qedittext
        print('[*back] precompute_queries()')
        prevBlock = self.win.blockSignals(True)
        self.precompute_feats()
        valid_cx = self.hs.get_valid_cxs()
        for qcx in valid_cx:
            print('[*back] query qcx=%r' % qcx)
            self.hs.query(qcx)
        self.win.blockSignals(prevBlock)
        print('[*back] Finished precomputing queries')

    # Option Actions
    @pyqtSlot(name='layout_figures')
    def layout_figures(self):
        print('[*back] layout')
        dlen = 1618
        if self.app is not None:
            app = self.app
            screen_rect = app.desktop().screenGeometry()
            width = screen_rect.width()
            height = screen_rect.height()
            dlen = np.sqrt(width ** 2 + height ** 2) / 1.618
        df2.present(num_rc=(2, 2), wh=dlen, wh_off=(0, 60))

    @pyqtSlot(name='dev_mode')
    def dev_mode(self):
        print('[*back] dev_mode (iteraction not working)')
        backend = self  # NOQA
        hs = self.hs    # NOQA
        devmode = True  # NOQA
        print(helpers.indent(str(hs), '[*back.hs] '))
        rrr()
        guitools.rrr()
        guifront.rrr()
        HotSpotter.rrr()
        viz.rrr()
        #if self.timer is not None:
            #self.timer.pause()
            #exec(helpers.ipython_execstr())
            #self.timer.start()

    # Help Actions
    @pyqtSlot(name='view_database_dir')
    def view_database_dir(self):
        self.hs.vdd()

    @pyqtSlot(name='view_computed_dir')
    def view_computed_dir(self):
        self.hs.vcd()

    @pyqtSlot(name='view_global_prefs')
    def view_global_dir(self):
        self.hs.vgd()
    #---

    @pyqtSlot(name='delete_computed_dir')
    def delete_computed_dir(self):
        self.hs.delete_computed_dir()

    @pyqtSlot(name='delete_global_prefs')
    def delete_global_prefs(self):
        self.hs.delete_global_prefs()


def make_main_window(hs=None, app=None):
    backend = MainWindowBackend(hs=hs)
    backend.app = app
    backend.win.show()
    backend.layout_figures()
    if app is not None:
        app.setActiveWindow(backend.win)
    return backend


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    app, is_root = guitools.init_qtapp()
    backend = guitools.make_main_window()
    win = backend.win
    ui = win.ui
    guitools.run_main_loop(app, is_root, backend)
