from __future__ import division, print_function
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

# Helper functions (should probably be moved into HotSpotter API)


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
        #return self.select_next_unannotated()
        self.selection = {'type_': 'gx', 'index': -1}
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


# SLOT DECORATOR
def slot_(*types):  # This is called at wrap time to get args
    'wrapper around pyqtslot decorator'
    DEBUG = True
    if DEBUG:
        # Wrap with debug statments

        def pyqtSlotWrapper(func):
            func_name = func.func_name
            print('[@back] Wrapping %r with slot_' % func.func_name)

            @pyqtSlot(*types, name=func.func_name)
            def slot_wrapper(self, *args, **kwargs):
                argstr_list = map(str, args)
                kwastr_list = ['%s=%s' % item for item in kwargs.iteritems()]
                argstr = ', '.join(argstr_list + kwastr_list)
                print('[**back.slot_] %s(%s)' % (func_name, argstr))
                #with helpers.Indenter():
                result = func(self, *args, **kwargs)
                print('[**back.slot_] Finished %s(%s)' % (func_name, argstr))
                return result

            slot_wrapper.func_name = func_name
            return slot_wrapper
    else:
        # Wrap wihout any debugging
        def pyqtSlotWrapper(func):
            func_name = func.func_name

            @pyqtSlot(*types, name=func.func_name)
            def slot_wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            slot_wrapper.func_name = func_name
            return slot_wrapper
    return pyqtSlotWrapper


# BLOCKING DECORATOR
def blocking(func):
    print('[@back] Wrapping %r with blocking' % func.func_name)

    def block_wrapper(self, *args, **kwargs):
        #print('[back] BLOCKING')
        #wasBlocked = self.blockSignals(True)
        wasBlocked_ = self.win.blockSignals(True)
        try:
            result = func(self, *args, **kwargs)
        except Exception as ex:
            #self.blockSignals(wasBlocked)
            self.win.blockSignals(wasBlocked_)
            print('Block wrapper caugt exception in %r' % func.func_name)
            print('self = %r' % self)
            print('*args = %r' % (args,))
            print('**kwargs = %r' % (kwargs,))
            print('ex = %r' % ex)
            self.user_info('Error in blocking ex=%r' % ex)
            raise
        #self.blockSignals(wasBlocked)
        self.win.blockSignals(wasBlocked_)
        #print('[back] UNBLOCKING')
        return result
    block_wrapper.func_name = func.func_name
    return block_wrapper


# DRAWING DECORATOR
def drawing(func):
    print('[@back] Wrapping %r with drawing' % func.func_name)

    def drawing_wrapper(self, *args, **kwargs):
        #print('[back] DRAWING')
        result = func(self, *args, **kwargs)
        #print('[back] DONE DRAWING')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()
        return result
    drawing_wrapper.func_name = func.func_name
    return drawing_wrapper


#------------------------
# Backend MainWindow Class
#------------------------
class MainWindowBackend(QtCore.QObject):
    'Sends and recieves signals to and from the frontend'
    #------------------------
    # Backend Signals
    #------------------------
    populateSignal = pyqtSignal(str, list, list, list, list)
    setEnabledSignal = pyqtSignal(bool)

    #------------------------
    # Constructor
    #------------------------
    def __init__(self, hs=None, app=None):
        super(MainWindowBackend, self).__init__()
        self.hs  = hs
        self.app = app
        self.current_res = None
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
        if hs is not None:
            self.connect_api(hs)

    #------------------------
    # Draw Functions
    #------------------------
    @drawing
    def show_splash(self, fnum=1, view='Nice', **kwargs):
        df2.figure(fnum=fnum, doclf=True, trueclf=True)
        viz.show_splash(fnum=fnum)
        df2.set_figtitle('%s View' % view)

    @drawing
    def show_image(self, gx, sel_cxs=[], figtitle='Image View', **kwargs):
        df2.figure(fnum=1, doclf=True, trueclf=True)
        cx_clicked_func = lambda cx: self.select_gx(gx, cx)
        viz.show_image(self.hs, gx, sel_cxs, cx_clicked_func,
                       fnum=1, figtitle=figtitle)

    @drawing
    def show_chip(self, cx, **kwargs):
        df2.figure(fnum=2, doclf=True, trueclf=True)
        INTERACTIVE_CHIPS = True  # This should always be True
        if INTERACTIVE_CHIPS:
            interact_fn = viz.show_chip_interaction
            interact_fn(self.hs, cx, fnum=2, figtitle='Chip View')
        else:
            viz.show_chip(self.hs, cx, fnum=2, figtitle='Chip View')

    @drawing
    def show_query_result(self, res, **kwargs):
        df2.figure(fnum=3, doclf=True, trueclf=True)

        def clicked_cid_fn(cid):
            cx_list = [self.hs.cid2_cx(cid)]
            return self.show_single_query(res, cx_list)
        ctrl_clicked_fn = drawing(viz.get_sv_from_cid_fn(self.hs, res.qcx))
        if self.hs.prefs.display_cfg.showanalysis:
            # Define callback for show_analysis
            res.show_analysis(self.hs, fnum=3, figtitle=' Analysis View',
                              clicked_cid_fn=clicked_cid_fn,
                              ctrl_clicked_cid_fn=ctrl_clicked_fn)
        else:
            res.show_top(self.hs, fnum=3, figtitle='Query View ',
                         clicked_cid_fn=clicked_cid_fn,
                         ctrl_clicked_cid_fn=ctrl_clicked_fn)

    @drawing
    def show_single_query(self, res, cx_list, **kwargs):
        # Define callback for show_analysis

        @drawing
        def clicked_cid_fn(cid):
            print('clicked me')
            pass
        df2.figure(fnum=4, doclf=True, trueclf=True)
        ctrl_clicked_fn = drawing(viz.get_sv_from_cid_fn(self.hs, res.qcx))
        res.show_analysis(self.hs, fnum=4, cx_list=cx_list, noshow_gt=True,
                          figtitle=' Result Inspection View',
                          clicked_cid_fn=clicked_cid_fn,
                          ctrl_clicked_fn=ctrl_clicked_fn)

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
        gx_list = self.hs.get_valid_gxs()
        datatup_list = self.hs.get_img_datatupe_list(gx_list)
        row_list = range(len(datatup_list))
        self.populateSignal.emit('image', col_headers, col_editable, row_list, datatup_list)

    def populate_result_table(self):
        print('[*back] populate_result_table()')
        res = self.current_res
        if res is None:
            print('[*back] no results available')
            return
        col_headers  = ['Rank', 'Matching Name', 'Chip ID',  'Confidence']
        col_editable = [False,             True,      True,         False]
        top_cxs = res.topN_cxs(self.hs, N='all')
        hs = self.hs
        qcx = res.qcx
        datatup_list = hs.get_res_datatup_list(top_cxs, res.cx2_score)
        # The ! mark is used for ascii sorting. TODO: can we work arround this?
        datatup_list = [('!Query Chip', hs.cx2_name(qcx), hs.cx2_cid(qcx), '---')] + datatup_list
        row_list = range(len(datatup_list))
        self.populateSignal.emit('res', col_headers, col_editable, row_list, datatup_list)

    def populate_chip_table(self):
        print('[*back] populate_chip_table()')
        col_headers  = ['Chip ID', 'Name', 'Image', '#GT']
        col_editable = [False,       True,   False, False]
        # Add User Properties to headers
        prop_dict = self.hs.tables.prop_dict
        prop_keys = prop_dict.keys()
        col_headers += prop_keys
        col_editable += [True] * len(prop_keys)
        # Populate table with valid image indexes
        cx_list = self.hs.get_valid_cxs()
        # Build lists of column values
        datatup_list = self.hs.get_chip_datatup_list(cx_list)
        # Define order of columns
        row_list = range(len(datatup_list))
        self.populateSignal.emit('chip', col_headers, col_editable, row_list, datatup_list)

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def user_info(self, *args, **kwargs):
        return guitools.user_info(self.win, *args, **kwargs)

    def user_input(self, *args, **kwargs):
        return guitools.user_input(self.win, *args, **kwargs)

    def user_option(self, *args, **kwargs):
        return guitools.user_option(self.win, *args, **kwargs)

    def get_work_directory(self, use_cache=True):
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

    #--------------------------------------------------------------------------
    # Misc Slots
    #--------------------------------------------------------------------------
    @slot_(str)
    def backend_print(self, msg):
        print(str(msg))

    @slot_()
    def clear_selection(self, **kwargs):
        self.selection = None
        self.show_splash(1, 'Image', dodraw=False)
        self.show_splash(2, 'Chip', dodraw=False)
        self.show_splash(3, 'Results', **kwargs)

    # Table Click -> Image Table
    @slot_(int)
    @blocking
    def select_gx(self, gx, cx=None, **kwargs):
        if cx is None:
            cxs = self.hs.gx2_cxs(gx)
            if len(cxs > 0):
                cx = cxs[0]
        if cx is None:
            self.show_splash(2, 'Chip', dodraw=False)
        else:
            self.show_chip(cx, dodraw=False)
        highlight_cxs = [] if cx is None else [cx]
        self.selection = {'type_': 'gx', 'index': gx, 'sub': cx}
        self.show_image(gx, highlight_cxs, **kwargs)

    # Table Click -> Chip Table
    @slot_(int)
    def select_cid(self, cid, **kwargs):
        cx = self.hs.cid2_cx(cid)
        gx = self.hs.cx2_gx(cx)
        self.select_gx(gx, cx=cx, **kwargs)

    # Table Click -> Chip Table
    @slot_(int)
    def select_res_cid(self, cid, **kwargs):
        cx = self.hs.cid2_cx(cid)
        gx = self.hs.cx2_gx(cx)
        self.select_gx(gx, cx=cx, dodraw=False, **kwargs)
        self.show_single_query(self.current_res, [cx], **kwargs)

    # Button Click -> Preferences Defaults
    @slot_()
    @blocking
    def default_preferences(self):
        # TODO: Propogate changes back to self.edit_prefs.ui
        self.hs.default_preferences()

    # Table Edit -> Change Chip Property
    @slot_(int, str, str)
    @blocking
    def change_chip_property(self, cid, key, val):
        key, val = map(str, (key, val))
        print('[*back] change_chip_property(%r, %r, %r)' % (cid, key, val))
        cx = self.hs.cid2_cx(cid)
        if key in ['Name', 'Matching Name']:
            self.hs.change_name(cx, val)
        else:
            self.hs.change_property(cx, key, val)
        self.populate_chip_table()
        self.populate_result_table()
        print('')

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------
    # File -> New Database
    @slot_()
    @blocking
    def new_database(self):
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
    @slot_()
    @blocking
    def open_database(self, db_dir=None):
        try:
            # Use the same args in a new (opened) database
            args = self.hs.args
            if db_dir is None:
                db_dir = guitools.select_directory('Select (or create) a database directory.')
            print('[*back] user selects database: ' + db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, db_dir=db_dir)
            hs.load(load_all=False)
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            self.connect_api(hs)
            self.layout_figures()
        except Exception as ex:
            print('aborting open database')
            print(ex)
            if self.hs.args.strict:
                raise
        print('')

    # File -> Save Database
    @slot_()
    @blocking
    def save_database(self):
        self.hs.save_database()

    # File -> Import Images
    @slot_()
    @blocking
    def import_images(self):
        print('[*back] import images')
        msg = 'Import specific files or whole directory?'
        title = 'Import Images'
        options = ['Files', 'Directory']
        reply = self.user_option(msg, title, options, False)
        if reply == 'Files':
            self.import_images_from_file()
        if reply == 'Directory':
            self.import_images_from_dir()

    # File -> Import Images From File
    @slot_()
    @blocking
    def import_images_from_file(self):
        fpath_list = guitools.select_images('Select image files to import')
        self.hs.add_images(fpath_list)
        self.populate_image_table()
        print('')

    # File -> Import Images From Directory
    @slot_()
    @blocking
    def import_images_from_dir(self):
        img_dpath = guitools.select_directory('Select directory with images in it')
        print('[*back] selected %r' % img_dpath)
        fpath_list = helpers.list_images(img_dpath, fullpath=True)
        self.hs.add_images(fpath_list)
        self.populate_image_table()
        print('')

    # File -> Quit
    @slot_()
    def quit(self):
        guitools.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------
    # Action -> New Chip Property
    @slot_()
    @blocking
    def new_prop(self):
        newprop = self.user_input('What is the new property name?')
        self.hs.add_property(newprop)
        self.populate_chip_table()
        self.populate_result_table()
        print(r'[/back] added newprop = %r' % newprop)
        print('')

    # Action -> Add ROI
    @slot_()
    @blocking
    def add_chip(self):
        gx = self.get_selected_gx()
        self.show_image(gx, figtitle='Image View - Select ROI (click two points)')
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        cx = self.hs.add_chip(gx, roi)
        self.populate_image_table()
        self.populate_chip_table()
        self.populate_result_table()
        self.select_gx(gx, cx)
        print('')

    # Action -> Query
    @slot_()
    @blocking
    def query(self, cid=None):
        #prevBlock = self.win.blockSignals(True)
        print('[**back] query(cid=%r)' % cid)
        cx = self.get_selected_cx() if cid is None else self.hs.cid2_cx(cid)
        print('[**back.query()] cx = %r)' % cx)
        if cx is None:
            self.user_info('Cannot query. No chip selected')
            return
        try:
            res = self.hs.query(cx)
        except Exception as ex:
            # TODO Catch actuall exceptions here
            print('[**back.query()] ex = %r' % ex)
            raise
        if isinstance(res, str):
            self.user_info(res)
            return
        self.current_res = res
        self.populate_result_table()
        print(r'[/back] finished query')
        print('')
        self.show_query_result(res)
        return res

    # Action -> Reselect ROI
    @slot_()
    @blocking
    def reselect_roi(self, **kwargs):
        print(r'[\back] reselect_roi()')
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot reselect ROI. No chip selected')
            return
        gx = self.hs.tables.cx2_gx[cx]
        self.show_image(gx, [cx], figtitle='Image View - ReSelect ROI (click two points)', **kwargs)
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        self.hs.change_roi(cx, roi)
        self.populate_image_table()
        self.populate_chip_table()
        self.populate_result_table()
        self.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected ROI = %r' % roi)
        print('')
        pass

    # Action -> Reselect ORI
    @slot_()
    @blocking
    def reselect_ori(self, **kwargs):
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot reselect orientation. No chip selected')
            return
        gx = self.hs.tables.cx2_gx[cx]
        self.show_image(gx, [cx], figtitle='Image View - Select Orientation (click two points)', **kwargs)
        theta = guitools.select_orientation()
        if theta is None:
            print('[back*] theta selection failed. Not adding')
            return
        self.hs.change_theta(cx, theta)
        self.populate_image_table()
        self.populate_chip_table()
        self.populate_result_table()
        self.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected theta=%r' % theta)
        print('')

    # Action -> Delete Chip
    @slot_()
    @blocking
    def delete_chip(self):
        cx = self.get_selected_cx()
        if cx is None:
            self.user_info('Cannot delete chip. No chip selected')
            return
        gx = self.hs.cx2_gx(cx)
        self.hs.delete_chip(cx)
        self.populate_image_table()
        self.populate_chip_table()
        self.populate_result_table()
        self.select_gx(gx)
        print('[back] deleted cx=%r\n' % cx)
        print('')

    # Action -> Next
    @slot_()
    @blocking
    def select_next(self):
        select_mode = 'in_order'  # 'unannotated'
        if select_mode == 'in_order':
            msg = select_next_in_order(self)
        elif select_mode == 'unannotated':
            msg = select_next_unannotated(self)
        else:
            raise Exception('uknown=%r' % select_mode)
        if msg is not None:
            self.user_info(msg)

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------
    # Batch -> Precompute Feats
    @slot_()
    @blocking
    def precompute_feats(self):
        #prevBlock = self.win.blockSignals(True)
        self.hs.update_samples()
        self.hs.refresh_features()
        #self.win.blockSignals(prevBlock)
        print('')

    # Batch -> Precompute Queries
    @slot_()
    @blocking
    def precompute_queries(self):
        # TODO:
        #http://stackoverflow.com/questions/15637768/
        # pyqt-how-to-capture-output-of-pythons-interpreter-
        # and-display-it-in-qedittext
        #prevBlock = self.win.blockSignals(True)
        import matching_functions as mf
        import DataStructures as ds
        import match_chips3 as mc3
        import sys
        self.precompute_feats()
        valid_cx = self.hs.get_valid_cxs()
        if self.hs.args.quiet:
            mc3.print_off()
            ds.print_off()
            mf.print_off()
        fmtstr = helpers.progress_str(len(valid_cx), '[back*] Query qcx=%r: ')
        for count, qcx in enumerate(valid_cx):
            sys.stdout.write(fmtstr % (qcx, count))
            self.hs.query(qcx, dochecks=False)
            if count % 100 == 0:
                sys.stdout.write('\n ...')
        sys.stdout.write('\n ...')
        mc3.print_on()
        ds.print_on()
        mf.print_on()
        print('')
        #self.win.blockSignals(prevBlock)

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------
    # Options -> Layout Figures
    @slot_()
    @blocking
    def layout_figures(self):
        if self.app is not None:
            app = self.app
            screen_rect = app.desktop().screenGeometry()
            width = screen_rect.width()
            height = screen_rect.height()
            dlen = np.sqrt(width ** 2 + height ** 2) / 1.618
        else:
            print('[*back] WARNING: cannot detect screen geometry')
            dlen = 1618
        df2.present(num_rc=(2, 3), wh=dlen, wh_off=(0, 60))

    # Options -> Edit Preferences
    @slot_()
    def edit_preferences(self):
        self.edit_prefs = self.hs.prefs.createQWidget()
        epw = self.edit_prefs
        epw.ui.defaultPrefsBUT.clicked.connect(self.default_preferences)
        query_uid = ''.join(self.hs.prefs.query_cfg.get_uid())
        print('[*back] query_uid = %s' % query_uid)
        print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------
    # Help -> View Directory Slots
    @slot_()
    def view_database_dir(self):
        self.hs.vdd()

    @slot_()
    def view_computed_dir(self):
        self.hs.vcd()

    @slot_()
    def view_global_dir(self):
        self.hs.vgd()

    # Help -> Delete Directory Slots
    @slot_()
    def delete_computed_dir(self):
        self.hs.delete_computed_dir()

    @slot_()
    def delete_global_prefs(self):
        self.hs.delete_global_prefs()

    # Help -> Developer Help
    @slot_()
    @blocking
    def dev_help(self):
        backend = self  # NOQA
        hs = self.hs    # NOQA
        devmode = True  # NOQA
        print(helpers.indent(str(hs), '[*back.hs] '))
        rrr()
        guitools.rrr()
        guifront.rrr()
        HotSpotter.rrr()
        viz.rrr()
        print(r'[\back] finished dev_help')
        #app = self.app
        #from PyQt4 import QtGui
        #QtGui.qApp.quit()
        #app.exit()  # Stop the main loop
        #app.quit()
        #if self.timer is not None:
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        #from IPython.lib.inputhook import enable_qt4
        #enable_qt4()
        execstr = helpers.ipython_execstr()
        #print(execstr)
        print('Debugging in IPython. IPython will break gui until you exit')
        exec(execstr)
        #self.timer.start()


# Creation function
def make_main_window(hs=None, app=None):
    print(r'[*back] make_main_window()')
    backend = MainWindowBackend(hs=hs)
    backend.app = app
    backend.win.show()
    backend.layout_figures()
    if app is not None:
        app.setActiveWindow(backend.win)
    print('[*back] Finished creating main win\n')
    return backend


# Main Test Script
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    app, is_root = guitools.init_qtapp()
    backend = guitools.make_main_window()
    win = backend.win
    ui = win.ui
    guitools.run_main_loop(app, is_root, backend)
