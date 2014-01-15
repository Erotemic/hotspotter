from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[back]')
from os.path import split, exists, join
# Qt
from PyQt4 import QtCore
from PyQt4.Qt import pyqtSignal
import numpy as np
# Hotspotter
import guifront
import guitools
import helpers
import fileio as io
import draw_func2 as df2
import vizualizations as viz
import interaction
import HotSpotter
from guitools import drawing, slot_
from guitools import backblocking as blocking

FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5)
viz.register_FNUMS(FNUMS)


# Helper functions (should probably be moved into HotSpotter API)


def select_next_unannotated(back):
    msg = 'err'
    if back.selection is None or back.selection['type_'] == 'gx':
        valid_gxs = back.hs.get_valid_gxs()
        has_chips = lambda gx: len(back.hs.gx2_cxs(gx)) > 0
        hascxs_list = map(has_chips, iter(valid_gxs))
        try:
            gx = valid_gxs[hascxs_list.index(False)]
            back.select_gx(gx)
            return
        except ValueError:
            msg = 'All images have detections. Excellent! '
    if back.selection is None or msg is not None and back.selection['type_'] == 'cx':
        valid_cxs = back.hs.get_valid_cxs()
        has_name = lambda cx: back.hs.cx2_name(cx) != '____'
        is_named = map(has_name, iter(valid_cxs))
        try:
            cx = valid_cxs[is_named.index(False)]
            cid = back.hs.tables.cx2_cid[cx]
            back.select_cid(cid)
            return
        except ValueError:
            msg = 'All chips are named. Awesome! '
    return msg


def select_next_in_order(back):
    if back.selection is None:
        # No selection
        #return back.select_next_unannotated()
        back.selection = {'type_': 'gx', 'index': -1}
    if back.selection['type_'] == 'gx':
        # Select next image
        gx = back.selection['index']
        gx2_gname = back.hs.tables.gx2_gname
        next_gx = gx + 1
        while next_gx < len(gx2_gname):
            if gx2_gname[next_gx] != '':
                back.select_gx(next_gx)
                break
            next_gx += 1
        return
    elif back.selection['type_'] == 'cx':
        # Select next chip
        cx = back.selection['index']
        cx2_cid = back.hs.tables.cx2_cid
        next_cx = cx + 1
        while next_cx < len(cx2_cid):
            cid = cx2_cid[next_cx]
            if cid != 0:
                back.select_cid(cid)
                break
            next_cx += 1
        return
    return 'end of the list'

aif_header = 'AIF'


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
    def __init__(back, hs=None, app=None):
        super(MainWindowBackend, back).__init__()
        back.hs  = hs
        back.app = app
        back.current_res = None
        back.timer = None
        kwargs_ = {'use_plot_widget': False}
        back.front = guifront.MainWindowFrontend(back=back, **kwargs_)
        back.selection = None
        #if kwargs_['use_plot_widget']:
            #df2.register_matplotlib_widget(back.front.plotWidget)
        df2.register_qt4_win(back.front)
        # Define default table headers
        back.imgtbl_headers   = ['Image Index', 'Image Name', '#Chips']
        back.imgtbl_editable  = []
        with_aif = True
        if with_aif:
            back.imgtbl_headers += [aif_header]
            back.imgtbl_editable += [aif_header]

        if hs.args.withexif:
            back.imgtbl_headers += ['EXIF']
        #
        back.chiptbl_headers  = ['Chip ID', 'Name', 'Image', '#GT', '#kpts', 'Theta', 'ROI (x, y, w, h)']
        back.chiptbl_editable = ['Name']
        #
        back.restbl_headers   = ['Rank', 'Confidence', 'Matching Name', 'Chip ID']
        back.restbl_editable  = ['Matching Name']
        # connect signals
        back.populateSignal.connect(back.front.populate_tbl)
        back.setEnabledSignal.connect(back.front.setEnabled)
        if hs is not None:
            back.connect_api(hs)

    #------------------------
    # Draw Functions
    #------------------------

    @drawing
    def show_splash(back, fnum, view='Nice', **kwargs):
        if df2.plt.fignum_exists(fnum):
            df2.figure(fnum=fnum, doclf=True, trueclf=True)
            viz.show_splash(fnum=fnum)
            df2.set_figtitle('%s View' % view)

    @drawing
    def show_image(back, gx, sel_cxs=[], figtitle='Image View', **kwargs):
        fnum = FNUMS['image']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, doclf=True, trueclf=True)
        cx_clicked_func = lambda cx: back.select_gx(gx, cx)
        viz.show_image(back.hs, gx, sel_cxs, cx_clicked_func,
                       fnum=fnum, figtitle=figtitle)
        if not did_exist:
            back.layout_figures()

    @drawing
    def show_chip(back, cx, **kwargs):
        fnum = FNUMS['chip']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, doclf=True, trueclf=True)
        INTERACTIVE_CHIPS = True  # This should always be True
        if INTERACTIVE_CHIPS:
            interact_fn = interaction.interact_chip
            interact_fn(back.hs, cx, fnum=fnum, figtitle='Chip View')
        else:
            viz.show_chip(back.hs, cx, fnum=fnum, figtitle='Chip View')
        if not did_exist:
            back.layout_figures()

    @drawing
    def show_query_result(back, res, tx=None, **kwargs):
        if tx is not None:
            fnum = FNUMS['inspect']
            did_exist = df2.plt.fignum_exists(fnum)
            # Interact with the tx\th top index
            res.interact_top_chipres(back.hs, tx)
        else:
            fnum = FNUMS['res']
            did_exist = df2.plt.fignum_exists(fnum)
            df2.figure(fnum=fnum, doclf=True, trueclf=True)
            if back.hs.prefs.display_cfg.showanalysis:
                # Define callback for show_analysis
                res.show_analysis(back.hs, fnum=fnum, figtitle=' Analysis View')
            else:
                res.show_top(back.hs, fnum=fnum, figtitle='Query View ')
        if not did_exist:
            back.layout_figures()

    @drawing
    def show_single_query(back, res, cx, **kwargs):
        # Define callback for show_analysis
        fnum = FNUMS['inspect']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, doclf=True, trueclf=True)
        interaction.interact_chipres(back.hs, cx, fnum=fnum)
        if not did_exist:
            back.layout_figures()

    #----------------------
    # Work Functions
    #----------------------
    def get_selected_gx(back):
        'selected image index'
        if back.selection is None:
            return None
        type_ = back.selection['type_']
        if type_ == 'gx':
            gx = back.selection['index']
        if type_ == 'cx':
            cx = back.selection['index']
            gx = back.hs.tables.cx2_gx(cx)
        return gx

    def get_selected_cx(back):
        'selected chip index'
        if back.selection is None:
            return None
        type_ = back.selection['type_']
        if type_ == 'cx':
            cx = back.selection['index']
        if type_ == 'gx':
            cx = back.selection['sub']
        return cx

    def update_window_title(back):
        if back.hs is None:
            title = 'Hotspotter - NULL database'
        if back.hs.dirs is None:
            title = 'Hotspotter - invalid database'
        else:
            db_dir = back.hs.dirs.db_dir
            db_name = split(db_dir)[1]
            title = 'Hotspotter - %r - %s' % (db_name, db_dir)
        back.front.setWindowTitle(title)

    def connect_api(back, hs):
        print('[*back] connect_api()')
        back.hs = hs
        if hs.tables is not None:
            back.populate_image_table()
            back.populate_chip_table()
            back.setEnabledSignal.emit(True)
            back.clear_selection()
            back.update_window_title()
            back.layout_figures()
        else:
            back.setEnabledSignal.emit(False)
        #back.database_loaded.emit()

    def populate_image_table(back):
        print('[*back] populate_image_table()')
        col_headers, col_editable = guitools.make_header_lists(back.imgtbl_headers,
                                                               back.imgtbl_editable)
        # Populate table with valid image indexes
        gx_list = back.hs.get_valid_gxs()
        datatup_list = back.hs.get_img_datatup_list(gx_list, header_order=col_headers)
        row_list = range(len(datatup_list))
        back.populateSignal.emit('image', col_headers, col_editable, row_list, datatup_list)

    def populate_chip_table(back):
        print('[*back] populate_chip_table()')
        # Add User Properties to headers
        prop_dict = back.hs.tables.prop_dict
        prop_keys = prop_dict.keys()
        col_headers, col_editable = guitools.make_header_lists(back.chiptbl_headers,
                                                               back.chiptbl_editable,
                                                               prop_keys)
        # Populate table with valid image indexes
        cx_list = back.hs.get_valid_cxs()
        # Build lists of column values
        datatup_list = back.hs.get_chip_datatup_list(cx_list, header_order=col_headers)
        # Define order of columns
        row_list = range(len(datatup_list))
        back.populateSignal.emit('chip', col_headers, col_editable, row_list, datatup_list)

    def populate_result_table(back):
        print('[*back] populate_result_table()')
        res = back.current_res
        if res is None:
            print('[*back] no results available')
            return
        col_headers, col_editable = guitools.make_header_lists(back.restbl_headers,
                                                               back.restbl_editable)
        top_cxs = res.topN_cxs(back.hs, N='all')
        hs = back.hs
        qcx = res.qcx
        datatup_list = hs.get_res_datatup_list(top_cxs, res.cx2_score,
                                               header_order=col_headers)
        # The ! mark is used for ascii sorting. TODO: can we work arround this?
        query_cols = {
            'Rank': '!Query',
            'Confidence': '---',
            'Matching Name': hs.cx2_name(qcx),
            'Chip ID': hs.cx2_cid(qcx),
        }
        querytup_list = [[query_cols[header] for header in col_headers]]
        datatup_list  = querytup_list + datatup_list
        row_list = range(len(datatup_list))
        back.populateSignal.emit('res', col_headers, col_editable, row_list, datatup_list)

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    # TODO: this code is duplicated in front
    def user_info(back, *args, **kwargs):
        return guitools.user_info(back.front, *args, **kwargs)

    def user_input(back, *args, **kwargs):
        return guitools.user_input(back.front, *args, **kwargs)

    def user_option(back, *args, **kwargs):
        return guitools.user_option(back.front, *args, **kwargs)

    def get_work_directory(back, use_cache=True):
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
            try_again = back.user_option(msg_try, 'get work dir failed', opt_try, False)
            if try_again == 'Try Again':
                return back.get_work_dir(use_cache)
        io.global_cache_write(cache_id, work_dir)
        return work_dir

    #--------------------------------------------------------------------------
    # Misc Slots
    #--------------------------------------------------------------------------
    @slot_(str)
    def backend_print(back, msg):
        print(str(msg))

    @slot_()
    def clear_selection(back, **kwargs):
        back.selection = None
        back.show_splash(FNUMS['image'], 'Image', dodraw=False)
        back.show_splash(FNUMS['chip'], 'Chip', dodraw=False)
        back.show_splash(FNUMS['res'], 'Results', **kwargs)

    # Table Click -> Image Table
    @slot_(int)
    @blocking
    def select_gx(back, gx, cx=None, **kwargs):
        autoselect_chips = False
        if autoselect_chips and cx is None:
            cxs = back.hs.gx2_cxs(gx)
            if len(cxs > 0):
                cx = cxs[0]
        if cx is None:
            back.show_splash(2, 'Chip', dodraw=False)
        else:
            back.show_chip(cx, dodraw=False)
        highlight_cxs = [] if cx is None else [cx]
        back.selection = {'type_': 'gx', 'index': gx, 'sub': cx}
        back.show_image(gx, highlight_cxs, **kwargs)

    # Table Click -> Chip Table
    @slot_(int)
    def select_cid(back, cid, **kwargs):
        cx = back.hs.cid2_cx(cid)
        gx = back.hs.cx2_gx(cx)
        back.select_gx(gx, cx=cx, **kwargs)

    # Table Click -> Chip Table
    @slot_(int)
    def select_res_cid(back, cid, **kwargs):
        cx = back.hs.cid2_cx(cid)
        gx = back.hs.cx2_gx(cx)
        back.select_gx(gx, cx=cx, dodraw=False, **kwargs)
        back.show_single_query(back.current_res, cx, **kwargs)

    # Button Click -> Preferences Defaults
    @slot_()
    @blocking
    def default_preferences(back):
        # TODO: Propogate changes back to back.edit_prefs.ui
        back.hs.default_preferences()
        back.hs.prefs.save()

    # RCOS TODO: These function should take the type of the variable as an
    # arugment as well

    # Table Edit -> Change Chip Property
    @slot_(int, str, str)
    @blocking
    def change_chip_property(back, cid, key, val):
        key, val = map(str, (key, val))
        print('[*back] change_chip_property(%r, %r, %r)' % (cid, key, val))
        cx = back.hs.cid2_cx(cid)
        if key in ['Name', 'Matching Name']:
            back.hs.change_name(cx, val)
        else:
            back.hs.change_property(cx, key, val)
        back.populate_chip_table()
        back.populate_result_table()
        print('')

    # Table Edit -> Change Image Property
    @slot_(int, str, bool)
    @blocking
    def change_image_property(back, gx, key, val):
        key, val = str(key), bool(val)
        print('[*back] change_img_property(%r, %r, %r)' % (gx, key, val))
        if key in [aif_header]:
            back.hs.change_aif(gx, val)
        back.populate_image_table()
        print('')

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------
    # File -> New Database
    @slot_()
    @blocking
    def new_database(back):
        new_db = back.user_input('Enter the new database name')
        msg_put = 'Where should I put %r?' % new_db
        opt_put = ['Choose Directory', 'My Work Dir']
        reply = back.user_option(msg_put, 'options', opt_put, True)
        if reply == opt_put[1]:
            put_dir = back.get_work_directory()
        elif reply == opt_put[0]:
            put_dir = guitools.select_directory('Select where to put the new database')
        elif reply is None:
            back.user_info('No Reply. Aborting new database')
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
            try_again = back.user_option(msg_try, title_try, opt_try, False)
            if try_again == 'Try Again':
                return back.new_database()
        print('[*back] valid new_db_dir = %r' % new_db_dir)
        helpers.ensurepath(new_db_dir)
        back.open_database(new_db_dir)

    # File -> Open Database
    @slot_()
    @blocking
    def open_database(back, db_dir=None):
        try:
            # Use the same args in a new (opened) database
            args = back.hs.args
            if db_dir is None:
                db_dir = guitools.select_directory('Select (or create) a database directory.')
            print('[*back] user selects database: ' + db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, db_dir=db_dir)
            hs.load(load_all=False)
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            back.connect_api(hs)
            #back.layout_figures()
        except Exception as ex:
            import traceback
            print(traceback.format_exc())
            back.user_info('Aborting open database')
            print('aborting open database')
            print(ex)
            if back.hs.args.strict:
                raise
        print('')

    # File -> Save Database
    @slot_()
    @blocking
    def save_database(back):
        back.hs.save_database()

    # File -> Import Images
    @slot_()
    @blocking
    def import_images(back):
        print('[*back] import images')
        msg = 'Import specific files or whole directory?'
        title = 'Import Images'
        options = ['Files', 'Directory']
        reply = back.user_option(msg, title, options, False)
        if reply == 'Files':
            back.import_images_from_file()
        if reply == 'Directory':
            back.import_images_from_dir()

    # File -> Import Images From File
    @slot_()
    @blocking
    def import_images_from_file(back):
        fpath_list = guitools.select_images('Select image files to import')
        back.hs.add_images(fpath_list)
        back.populate_image_table()
        print('')

    # File -> Import Images From Directory
    @slot_()
    @blocking
    def import_images_from_dir(back):
        img_dpath = guitools.select_directory('Select directory with images in it')
        print('[*back] selected %r' % img_dpath)
        fpath_list = helpers.list_images(img_dpath, fullpath=True)
        back.hs.add_images(fpath_list)
        back.populate_image_table()
        print('')

    # File -> Quit
    @slot_()
    def quit(back):
        guitools.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------
    # Action -> New Chip Property
    @slot_()
    @blocking
    def new_prop(back):
        newprop = back.user_input('What is the new property name?')
        back.hs.add_property(newprop)
        back.populate_chip_table()
        back.populate_result_table()
        print(r'[/back] added newprop = %r' % newprop)
        print('')

    # Action -> Add ROI
    @slot_()
    @blocking
    def add_chip(back):
        gx = back.get_selected_gx()
        back.show_image(gx, figtitle='Image View - Select ROI (click two points)')
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        cx = back.hs.add_chip(gx, roi)
        back.populate_image_table()
        back.populate_chip_table()
        back.populate_result_table()
        back.select_gx(gx, cx)
        print('')

    # Action -> Query
    @slot_()
    @blocking
    def query(back, cid=None, tx=None):
        #prevBlock = back.front.blockSignals(True)
        print('[**back] query(cid=%r)' % cid)
        cx = back.get_selected_cx() if cid is None else back.hs.cid2_cx(cid)
        print('[**back.query()] cx = %r)' % cx)
        if cx is None:
            back.user_info('Cannot query. No chip selected')
            return
        try:
            res = back.hs.query(cx)
        except Exception as ex:
            # TODO Catch actuall exceptions here
            print('[**back.query()] ex = %r' % ex)
            raise
        if isinstance(res, str):
            back.user_info(res)
            return
        back.current_res = res
        back.populate_result_table()
        print(r'[/back] finished query')
        print('')
        back.show_query_result(res, tx)
        return res

    # Action -> Reselect ROI
    @slot_()
    @blocking
    def reselect_roi(back, **kwargs):
        print(r'[\back] reselect_roi()')
        cx = back.get_selected_cx()
        if cx is None:
            back.user_info('Cannot reselect ROI. No chip selected')
            return
        gx = back.hs.tables.cx2_gx[cx]
        back.show_image(gx, [cx], figtitle='Image View - ReSelect ROI (click two points)', **kwargs)
        roi = guitools.select_roi()
        if roi is None:
            print('[back*] roiselection failed. Not adding')
            return
        back.hs.change_roi(cx, roi)
        back.populate_image_table()
        back.populate_chip_table()
        back.populate_result_table()
        back.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected ROI = %r' % roi)
        print('')
        pass

    # Action -> Reselect ORI
    @slot_()
    @blocking
    def reselect_ori(back, **kwargs):
        cx = back.get_selected_cx()
        if cx is None:
            back.user_info('Cannot reselect orientation. No chip selected')
            return
        gx = back.hs.tables.cx2_gx[cx]
        back.show_image(gx, [cx], figtitle='Image View - Select Orientation (click two points)', **kwargs)
        theta = guitools.select_orientation()
        if theta is None:
            print('[back*] theta selection failed. Not adding')
            return
        back.hs.change_theta(cx, theta)
        back.populate_image_table()
        back.populate_chip_table()
        back.populate_result_table()
        back.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected theta=%r' % theta)
        print('')

    # Action -> Delete Chip
    @slot_()
    @blocking
    def delete_chip(back):
        # RCOS TODO: Are you sure?
        cx = back.get_selected_cx()
        if cx is None:
            back.user_info('Cannot delete chip. No chip selected')
            return
        gx = back.hs.cx2_gx(cx)
        back.hs.delete_chip(cx)
        back.populate_image_table()
        back.populate_chip_table()
        back.populate_result_table()
        back.select_gx(gx)
        print('[back] deleted cx=%r\n' % cx)
        print('')

    # Action -> Next
    @slot_()
    @blocking
    def select_next(back):
        select_mode = 'in_order'  # 'unannotated'
        if select_mode == 'in_order':
            msg = select_next_in_order(back)
        elif select_mode == 'unannotated':
            msg = select_next_unannotated(back)
        else:
            raise Exception('uknown=%r' % select_mode)
        if msg is not None:
            back.user_info(msg)

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------
    # Batch -> Precompute Feats
    @slot_()
    @blocking
    def precompute_feats(back):
        #prevBlock = back.front.blockSignals(True)
        back.hs.update_samples()
        back.hs.refresh_features()
        #back.front.blockSignals(prevBlock)
        back.populate_chip_table()
        print('')

    # Batch -> Precompute Queries
    @slot_()
    @blocking
    def precompute_queries(back):
        # TODO:
        #http://stackoverflow.com/questions/15637768/
        # pyqt-how-to-capture-output-of-pythons-interpreter-
        # and-display-it-in-qedittext
        #prevBlock = back.front.blockSignals(True)
        import matching_functions as mf
        import DataStructures as ds
        import match_chips3 as mc3
        import sys
        back.precompute_feats()
        valid_cx = back.hs.get_valid_cxs()
        if back.hs.args.quiet:
            mc3.print_off()
            ds.print_off()
            mf.print_off()
        fmtstr = helpers.progress_str(len(valid_cx), '[back*] Query qcx=%r: ')
        for count, qcx in enumerate(valid_cx):
            sys.stdout.write(fmtstr % (qcx, count))
            back.hs.query(qcx, dochecks=False)
            if count % 100 == 0:
                sys.stdout.write('\n ...')
        sys.stdout.write('\n ...')
        mc3.print_on()
        ds.print_on()
        mf.print_on()
        print('')
        #back.front.blockSignals(prevBlock)

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------
    # Options -> Layout Figures
    @slot_(rundbg=True)
    @blocking
    def layout_figures(back):
        print('[back] layout_figures')
        nCols = 3
        nRows = 2
        if back.app is None:
            print('[*back] WARNING: cannot detect screen geometry')
            dlen = 1618
        else:
            app = back.app
            screen_rect = app.desktop().screenGeometry()
            width  = screen_rect.width()
            height = screen_rect.height()
            dlen = np.sqrt(width ** 2 + height ** 2) / 1.618
        df2.present(num_rc=(nRows, nCols), wh=dlen, wh_off=(0, 60))

    # Options -> Edit Preferences
    @slot_()
    def edit_preferences(back):
        back.edit_prefs = back.hs.prefs.createQWidget()
        epw = back.edit_prefs
        epw.ui.defaultPrefsBUT.clicked.connect(back.default_preferences)
        query_uid = ''.join(back.hs.prefs.query_cfg.get_uid())
        print('[*back] query_uid = %s' % query_uid)
        print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------
    # Help -> View Directory Slots
    @slot_()
    def view_database_dir(back):
        back.hs.vdd()

    @slot_()
    def view_computed_dir(back):
        back.hs.vcd()

    @slot_()
    def view_global_dir(back):
        back.hs.vgd()

    # Help -> Delete Directory Slots
    @slot_()
    def delete_cache(back):
        # RCOS TODO: Are you sure?
        ans = guitools.user_option(back, 'Are you sure you want to delete cache?')
        if ans != 'Yes':
            return
        df2.close_all_figures()
        back.hs.delete_cache()

    @slot_()
    def delete_global_prefs(back):
        # RCOS TODO: Are you sure?
        df2.close_all_figures()
        back.hs.delete_global_prefs()

    @slot_()
    def delete_queryresults_dir(back):
        # RCOS TODO: Are you sure?
        df2.close_all_figures()
        back.hs.delete_queryresults_dir()

    # Help -> Developer Help
    @slot_()
    @blocking
    def dev_mode(back):
        steal_again = back.front.return_stdout()
        hs = back.hs    # NOQA
        devmode = True  # NOQA
        print(helpers.indent(str(hs), '[*back.hs] '))
        rrr()
        print(r'[\back] finished dev_help')
        #app = back.app
        #from PyQt4 import QtGui
        #QtGui.qApp.quit()
        #app.exit()  # Stop the main loop
        #app.quit()
        #if back.timer is not None:
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        #from IPython.lib.inputhook import enable_qt4
        #enable_qt4()
        execstr = helpers.ipython_execstr()
        #print(execstr)
        print('Debugging in IPython. IPython will break gui until you exit')
        exec(execstr)
        if steal_again:
            back.front.steal_stdout()
        #back.timer.start()

    # Help -> Developer Reload
    @slot_()
    @blocking
    def dev_reload(back):
        import dev
        dev.dev_reload()
        df2.unregister_qt4_win('all')
        df2.register_qt4_win(back.front)

    def show(back):
        back.front.show()


# Creation function
def make_main_window(hs=None, app=None):
    #printDBG(r'[*back] make_main_window()')
    back = MainWindowBackend(hs=hs)
    back.app = app
    if not hs.args.nogui:
        back.show()
        back.layout_figures()
        if app is not None:
            app.setActiveWindow(back.front)
    #print('[*back] Finished creating main front\n')
    return back


# Main Test Script
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    app, is_root = guitools.init_qtapp()
    back = guitools.make_main_window()
    front = back.front
    ui = front.ui
    guitools.run_main_loop(app, is_root, back)
