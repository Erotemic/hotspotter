from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[back]')
# Python
from os.path import split, exists, join
# Qt
from PyQt4 import QtCore
from PyQt4.Qt import pyqtSignal
# Science
import numpy as np
# Hotspotter
import guifront
import guitools
from guitools import drawing, slot_
from guitools import backblocking as blocking
from hscom import helpers as util
from hscom import fileio as io
from hscom import params
from hsviz import draw_func2 as df2
from hsviz import viz
from hsviz import interact
from hotspotter import HotSpotterAPI

FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)
viz.register_FNUMS(FNUMS)


# Helper functions (should probably be moved into HotSpotter API)


def select_next_unannotated(back):
    # FIXME THIS FUNCTION IS STUPID MESSY (and probably broken)
    msg = 'err'
    selection_exists = back.selection is None
    if selection_exists or back.selection['type_'] == 'gx':
        valid_gxs = back.hs.get_valid_gxs()
        has_chips = lambda gx: len(back.hs.gx2_cxs(gx)) > 0
        hascxs_list = map(has_chips, iter(valid_gxs))
        try:
            gx = valid_gxs[hascxs_list.index(False)]
            back.select_gx(gx)
            return
        except ValueError:
            msg = 'All images have detections. Excellent! '

    was_err = msg is not None
    cx_is_selected = selection_exists and back.selection['type_'] == 'cx'
    if selection_exists or (was_err and cx_is_selected):
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


# Creation function
def make_main_window(app=None, hs=None):
    #printDBG(r'[*back] make_main_window()')
    back = MainWindowBackend(app=app, hs=hs)
    if hs is None or not params.args.nogui:
        back.show()
        back.layout_figures()
        if app is not None:
            app.setActiveWindow(back.front)
    #print('[*back] Finished creating main front\n')
    return back


def _dev_reload(back):
    from hsdev import dev_reload
    dev_reload.reload_all_modules()
    df2.unregister_qt4_win('all')
    df2.register_qt4_win(back.front)
    back.populate_tables()


def _user_select_new_dbdir(back):
    'script for new database user interaction'
    try:
        # Ask the user what to call the new database
        new_db = back.user_input('Enter the new database name')
        # Return on cancel
        if new_db is None:
            raise StopIteration('Canceled')
        # Ask the user where to put the new database
        msg_put = 'Where should I put %r?' % new_db
        opt_put = ['Choose Directory', 'My Work Dir']
        reply = back.user_option(msg_put, 'options', opt_put, True)
        if reply == opt_put[1]:
            put_dir = back.get_work_directory()
        elif reply == opt_put[0]:
            put_dir = guitools.select_directory(
                'Select where to put the new database')
        else:
            raise StopIteration('Canceled')
        new_dbdir = join(put_dir, new_db)
        if not exists(put_dir):
            raise ValueError('Directory %r does not exist.' % put_dir)
        elif exists(new_dbdir):
            raise ValueError('New DB %r already exists.' % new_dbdir)
        return new_dbdir
    except ValueError as ex:
        opt_try = ['Try Again']
        title_try = 'New Database Failed'
        try_again = back.user_option(str(ex), title_try, opt_try, False)
        if try_again == 'Try Again':
            return _user_select_new_dbdir(back)
    except StopIteration as ex:
        pass
    return None


#------------------------
# Backend MainWindow Class
#------------------------
class MainWindowBackend(QtCore.QObject):
    '''
    Sends and recieves signals to and from the frontend
    '''
    # Backend Signals
    populateSignal = pyqtSignal(str, list, list, list, list)
    setEnabledSignal = pyqtSignal(bool)

    #------------------------
    # Constructor
    #------------------------
    def __init__(back, app=None, hs=None):
        super(MainWindowBackend, back).__init__()
        back.current_res = None
        back.timer = None
        back.selection = None

        # A map from short internal headers to fancy headers seen by the user
        back.fancy_headers = {
            'gx':         'Image Index',
            'nx':         'Name Index',
            'cid':        'Chip ID',
            'aif':        'All Detected',
            'gname':      'Image Name',
            'nCxs':       '#Chips',
            'name':       'Name',
            'nGt':        '#GT',
            'nKpts':      '#Kpts',
            'theta':      'Theta',
            'roi':        'ROI (x, y, w, h)',
            'rank':       'Rank',
            'score':      'Confidence',
            'match_name': 'Matching Name',
        }
        back.reverse_fancy = {v: k for (k, v) in back.fancy_headers.items()}

        # A list of default internal headers to display
        back.table_headers = {
            'gxs':  ['gx', 'gname', 'nCxs', 'aif'],
            'cxs':  ['cid', 'name', 'gname', 'nGt', 'nKpts', 'theta'],
            'nxs':  ['nx', 'name', 'nCxs'],
            'res':  ['rank', 'score', 'name', 'cid']
        }

        # Lists internal headers whos items are editable
        back.table_editable = {
            'gxs':  [],
            'cxs':  ['name'],
            'nxs':  ['name'],
            'res':  ['name'],
        }

        # connect signals and other objects
        back.hs  = hs
        back.app = app
        back.front = guifront.MainWindowFrontend(back=back)
        df2.register_qt4_win(back.front)
        back.populateSignal.connect(back.front.populate_tbl)
        back.setEnabledSignal.connect(back.front.setEnabled)
        if hs is not None:
            back.connect_api(hs)

    #------------------------
    # Draw Functions
    #------------------------

    def show(back):
        back.front.show()

    @drawing
    @profile
    def show_splash(back, fnum, view='Nice', **kwargs):
        if df2.plt.fignum_exists(fnum):
            df2.figure(fnum=fnum, docla=True, doclf=True)
            viz.show_splash(fnum=fnum)
            df2.set_figtitle('%s View' % view)

    def _layout_figures_if(back, did_exist):
        #back._layout_figures_if(did_exist)
        pass

    @drawing
    @profile
    def show_image(back, gx, sel_cxs=[], figtitle='Image View', **kwargs):
        fnum = FNUMS['image']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, docla=True, doclf=True)
        interact.interact_image(back.hs, gx, sel_cxs, back.select_cx,
                                fnum=fnum, figtitle=figtitle)
        back._layout_figures_if(did_exist)

    @drawing
    @profile
    def show_chip(back, cx, **kwargs):
        fnum = FNUMS['chip']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, docla=True, doclf=True)
        INTERACTIVE_CHIPS = True  # This should always be True
        if INTERACTIVE_CHIPS:
            interact_fn = interact.interact_chip
            interact_fn(back.hs, cx, fnum=fnum, figtitle='Chip View')
        else:
            viz.show_chip(back.hs, cx, fnum=fnum, figtitle='Chip View')
        back._layout_figures_if(did_exist)

    @drawing
    @profile
    def show_query_result(back, res, tx=None, **kwargs):
        if tx is not None:
            fnum = FNUMS['inspect']
            did_exist = df2.plt.fignum_exists(fnum)
            # Interact with the tx\th top index
            res.interact_top_chipres(back.hs, tx)
        else:
            fnum = FNUMS['res']
            did_exist = df2.plt.fignum_exists(fnum)
            df2.figure(fnum=fnum, docla=True, doclf=True)
            if back.hs.prefs.display_cfg.showanalysis:
                # Define callback for show_analysis
                res.show_analysis(back.hs, fnum=fnum, figtitle=' Analysis View')
            else:
                res.show_top(back.hs, fnum=fnum, figtitle='Query View ')
        back._layout_figures_if(did_exist)

    @drawing
    @profile
    def show_single_query(back, res, cx, **kwargs):
        # Define callback for show_analysis
        fnum = FNUMS['inspect']
        did_exist = df2.plt.fignum_exists(fnum)
        df2.figure(fnum=fnum, docla=True, doclf=True)
        interact.interact_chipres(back.hs, res, cx=cx, fnum=fnum)
        back._layout_figures_if(did_exist)

    @drawing
    @profile
    def show_nx(back, nx, sel_cxs=[], **kwargs):
        # Define callback for show_analysis
        fnum = FNUMS['name']
        df2.figure(fnum=fnum, docla=True, doclf=True)
        interact.interact_name(back.hs, nx, sel_cxs, back.select_cx,
                               fnum=fnum)

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

    def get_selected_cx(back, cid=None):
        'selected chip index'
        if cid is not None:
            try:
                cx = back.hs.cid2_cx(cid)
                return cx
            except IndexError as ex:
                print(ex)
                msg = 'Query qcid=%d does not exist / is invalid' % cid
                raise AssertionError(msg)
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
            hs.register_backend(back)
            back.populate_tables(res=False)
            back.setEnabledSignal.emit(True)
            back.clear_selection()
            back.update_window_title()
            back.layout_figures()
        else:
            back.setEnabledSignal.emit(False)
        #back.database_loaded.emit()

    #--------------------------------------------------------------------------
    # Populate functions
    #--------------------------------------------------------------------------

    @profile
    def _populate_table(back, tblname, extra_cols={},
                        index_list=None, prefix_cols=[]):
        print('[*back] _populate_table(%r)' % tblname)
        headers = back.table_headers[tblname]
        editable = back.table_editable[tblname]
        if tblname == 'cxs':  # in ['cxs', 'res']: TODO props in restable
            prop_keys = back.hs.tables.prop_dict.keys()
        else:
            prop_keys = []
        col_headers, col_editable = guitools.make_header_lists(headers,
                                                               editable,
                                                               prop_keys)
        if index_list is None:
            index_list = back.hs.get_valid_indexes(tblname)
        # Prefix datatup
        prefix_datatup = [[prefix_col.get(header, 'error')
                           for header in col_headers]
                          for prefix_col in prefix_cols]
        body_datatup = back.hs.get_datatup_list(tblname, index_list,
                                                col_headers, extra_cols)
        datatup_list = prefix_datatup + body_datatup
        row_list = range(len(datatup_list))
        # Populate with fancy headers.
        col_fancyheaders = [back.fancy_headers[key]
                            if key in back.fancy_headers else key
                            for key in col_headers]
        back.populateSignal.emit(tblname, col_fancyheaders, col_editable,
                                 row_list, datatup_list)

    def populate_image_table(back, **kwargs):
        back._populate_table('gxs', **kwargs)

    def populate_name_table(back, **kwargs):
        back._populate_table('nxs', **kwargs)

    def populate_chip_table(back, **kwargs):
        back._populate_table('cxs', **kwargs)

    def populate_result_table(back, **kwargs):
        res = back.current_res
        if res is None:
            # Clear the table instead
            print('[*back] no results available')
            back._populate_table('res', index_list=[])
            return
        top_cxs = res.topN_cxs(back.hs, N='all')
        qcx = res.qcx
        # The ! mark is used for ascii sorting. TODO: can we work arround this?
        prefix_cols = [{'rank': '!Query',
                        'score': '---',
                        'name': back.hs.cx2_name(qcx),
                        'cid': back.hs.cx2_cid(qcx), }]
        extra_cols = {
            'score':  lambda cxs:  [res.cx2_score[cx] for cx in iter(cxs)],
        }
        back._populate_table('res', index_list=top_cxs,
                             prefix_cols=prefix_cols,
                             extra_cols=extra_cols,
                             **kwargs)

    def populate_tables(back, image=True, chip=True, name=True, res=True):
        if image:
            back.populate_image_table()
        if chip:
            back.populate_chip_table()
        if name:
            back.populate_name_table()
        if res:
            back.populate_result_table()

    def append_header(back, tblname, header, editable=False):
        try:
            pos = back.table_headers[tblname].index(header)
            print('[back] %s_TBL already has header=%r at pos=%d' %
                  (tblname, header, pos))
        except ValueError:
            back.table_headers[tblname].append(header)

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def user_info(back, *args, **kwargs):
        # TODO: this code is duplicated in front
        return guitools.user_info(back.front, *args, **kwargs)

    def user_input(back, *args, **kwargs):
        return guitools.user_input(back.front, *args, **kwargs)

    def user_option(back, *args, **kwargs):
        return guitools._user_option(back.front, *args, **kwargs)

    def get_work_directory(back):
        return params.get_workdir()

    def get_work_directory2(back, use_cache=True):
        # TODO: This should go in api (or higher level main?)
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
            try_again = back.user_option(msg_try, 'get work dir failed',
                                         opt_try, False)
            if try_again == 'Try Again':
                return back.get_work_dir(use_cache)
        io.global_cache_write(cache_id, work_dir)
        return work_dir

    def user_select_new_dbdir(back):
        return _user_select_new_dbdir(back)

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    @slot_(int)
    @blocking
    @profile
    def select_gx(back, gx, cx=None, show=True, **kwargs):
        # Table Click -> Image Table
        autoselect_chips = False
        if autoselect_chips and cx is None:
            cxs = back.hs.gx2_cxs(gx)
            if len(cxs > 0):
                cx = cxs[0]
        sel_cxs = [] if cx is None else [cx]
        back.selection = {'type_': 'gx', 'index': gx, 'sub': cx}
        if show:
            if cx is None:
                back.show_splash(2, 'Chip', dodraw=False)
            else:
                back.show_chip(cx, dodraw=False)
            back.show_image(gx, sel_cxs, **kwargs)

    @slot_(int)
    def select_cid(back, cid, **kwargs):
        # Table Click -> Chip Table
        cx = back.hs.cid2_cx(cid)
        gx = back.hs.cx2_gx(cx)
        back.select_gx(gx, cx=cx, **kwargs)

    @slot_(int)
    def select_cx(back, cx, **kwargs):
        gx = back.hs.cx2_gx(cx)
        back.select_gx(gx, cx=cx, **kwargs)

    @slot_(int)
    def select_nx(back, nx):
        back.show_nx(nx)

    @slot_(str)
    def select_name(back, name):
        name = str(name)
        nx = np.where(back.hs.tables.nx2_name == name)[0]
        back.select_nx(nx)

    @slot_(int)
    def select_res_cid(back, cid, **kwargs):
        # Table Click -> Chip Table
        cx = back.hs.cid2_cx(cid)
        gx = back.hs.cx2_gx(cx)
        back.select_gx(gx, cx=cx, dodraw=False, **kwargs)
        back.show_single_query(back.current_res, cx, **kwargs)

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

    @slot_()
    @blocking
    def default_preferences(back):
        # Button Click -> Preferences Defaults
        # TODO: Propogate changes back to back.edit_prefs.ui
        back.hs.default_preferences()
        back.hs.prefs.save()

    @slot_(int, str, str)
    @blocking
    @profile
    def change_chip_property(back, cid, key, val):
        # Table Edit -> Change Chip Property
        # RCOS TODO: These function should take the type of the variable as an
        # arugment as well. (Guifront tries to automatically interpret the
        # variable type by its value and it will get stuck on things like
        # 'True'. Is that a string or a bool? I don't know. We should tell it.)
        key, val = map(str, (key, val))
        print('[*back] change_chip_property(%r, %r, %r)' % (cid, key, val))
        cx = back.hs.cid2_cx(cid)
        if key in ['name', 'matching_name']:
            back.hs.change_name(cx, val)
        else:
            back.hs.change_property(cx, key, val)
        back.populate_tables(image=False)
        print('')

    @slot_(int, str, str)
    @blocking
    @profile
    def alias_name(back, nx, key, val):
        key, val = map(str, (key, val))
        print('[*back] alias_name(%r, %r, %r)' % (nx, key, val))
        if key in ['name']:
            # TODO: Add option to change name if alias fails
            back.hs.alias_name(nx, val)
        back.populate_tables(image=False)
        print('')

    @slot_(int, str, bool)
    @blocking
    def change_image_property(back, gx, key, val):
        # Table Edit -> Change Image Property
        key, val = str(key), bool(val)
        print('[*back] change_img_property(%r, %r, %r)' % (gx, key, val))
        if key in ['aif']:
            back.hs.change_aif(gx, val)
        back.populate_image_table()
        print('')

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------

    @slot_()
    @blocking
    def new_database(back, new_dbdir=None):
        # File -> New Database
        if new_dbdir is None:
            new_dbdir = back.user_select_new_dbdir()
        if new_dbdir is not None:
            print('[*back] valid new_dbdir = %r' % new_dbdir)
            util.ensurepath(new_dbdir)
            back.open_database(new_dbdir)
        else:
            print('[*back] abort new database()')

    @slot_()
    @blocking
    def open_database(back, db_dir=None):
        # File -> Open Database
        try:
            # Use the same args in a new (opened) database
            args = params.args
            #args = back.params.args
            if db_dir is None:
                msg = 'Select (or create) a database directory.'
                db_dir = guitools.select_directory(msg)
            print('[*back] user selects database: ' + db_dir)
            # Try and load db
            if args is not None:
                args.dbdir = db_dir
            hs = HotSpotterAPI.HotSpotter(args=args, db_dir=db_dir)
            hs.load(load_all=False)
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            back.connect_api(hs)
            #back.layout_figures()
        except Exception as ex:
            import traceback
            import sys
            print(traceback.format_exc())
            back.user_info('Aborting open database')
            print('aborting open database')
            print(ex)
            if '--strict' in sys.argv:
                raise
        print('')
        return hs

    @slot_()
    @blocking
    def save_database(back):
        # File -> Save Database
        back.hs.save_database()

    @slot_()
    @blocking
    def import_images(back):
        # File -> Import Images
        print('[*back] import images')
        msg = 'Import specific files or whole directory?'
        title = 'Import Images'
        options = ['Files', 'Directory']
        reply = back.user_option(msg, title, options, False)
        if reply == 'Files':
            back.import_images_from_file()
        if reply == 'Directory':
            back.import_images_from_dir()

    @slot_()
    @blocking
    def import_images_from_file(back):
        # File -> Import Images From File
        fpath_list = guitools.select_images('Select image files to import')
        back.hs.add_images(fpath_list)
        back.populate_image_table()
        print('')

    @slot_()
    @blocking
    def import_images_from_dir(back):
        # File -> Import Images From Directory
        msg = 'Select directory with images in it'
        img_dpath = guitools.select_directory(msg)
        print('[*back] selected %r' % img_dpath)
        fpath_list = util.list_images(img_dpath, fullpath=True)
        back.hs.add_images(fpath_list)
        back.populate_image_table()
        print('')

    @slot_()
    def quit(back):
        # File -> Quit
        guitools.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @slot_()
    @blocking
    def new_prop(back):
        # Action -> New Chip Property
        newprop = back.user_input('What is the new property name?')
        back.hs.add_property(newprop)
        back.populate_chip_table()
        back.populate_result_table()
        print(r'[/back] added newprop = %r' % newprop)
        print('')

    @slot_()
    @blocking
    @profile
    def add_chip(back, gx=None, roi=None):
        # Action -> Add ROI
        if gx is None:
            gx = back.get_selected_gx()
        if roi is None:
            figtitle = 'Image View - Select ROI (click two points)'
            back.show_image(gx, figtitle=figtitle)
            roi = guitools.select_roi()
            if roi is None:
                print('[back*] roiselection failed. Not adding')
                return
        cx = back.hs.add_chip(gx, roi)  # NOQA
        back.populate_tables()
        # RCOS TODO: Autoselect should be an option
        #back.select_gx(gx, cx)
        back.select_gx(gx)
        print('')
        cid = back.hs.cx2_cid(cx)
        return cid

    @slot_()
    @blocking
    @profile
    def query(back, cid=None, tx=None, **kwargs):
        # Action -> Query

        with util.Indent('[back.prequery]'):
            print('[back] query(cid=%r, %r)' % (cid, kwargs))
            cx = back.get_selected_cx(cid)
            print('[back] cx = %r' % cx)
            if cx is None:
                back.user_info('Cannot query. No chip selected')
                return
        with util.Indent('[back.query]'):
            try:
                res = back.hs.query(cx, **kwargs)
            except Exception as ex:
                # TODO Catch actually exceptions here
                print('[back] ex = %r' % ex)
                raise
        with util.Indent('[back.postquery]'):
            if isinstance(res, str):
                back.user_info(res)
                return
            back.current_res = res
            back.populate_result_table()
            print(r'[back] finished query')
            print('')
            # Show results against test chip index (tx)
            back.show_query_result(res, tx)
        return res

    @slot_()
    @blocking
    @profile
    def reselect_roi(back, cid=None, roi=None, **kwargs):
        # Action -> Reselect ROI
        print(r'[\back] reselect_roi()')
        cx = back.get_selected_cx(cid)
        if cx is None:
            back.user_info('Cannot reselect ROI. No chip selected')
            return
        gx = back.hs.tables.cx2_gx[cx]
        if roi is None:
            figtitle = 'Image View - ReSelect ROI (click two points)'
            back.show_image(gx, [cx], figtitle=figtitle, **kwargs)
            roi = guitools.select_roi()
            if roi is None:
                print('[back*] roiselection failed. Not changing')
                return
        back.hs.change_roi(cx, roi)
        back.populate_tables()
        back.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected ROI = %r' % roi)
        print('')
        pass

    @slot_()
    @blocking
    @profile
    def reselect_ori(back, cid=None, theta=None, **kwargs):
        # Action -> Reselect ORI
        cx = back.get_selected_cx(cid)
        if cx is None:
            back.user_info('Cannot reselect orientation. No chip selected')
            return
        gx = back.hs.tables.cx2_gx[cx]
        if theta is None:
            figtitle = 'Image View - Select Orientation (click two points)'
            back.show_image(gx, [cx], figtitle=figtitle, **kwargs)
            theta = guitools.select_orientation()
            if theta is None:
                print('[back*] theta selection failed. Not changing')
                return
        back.hs.change_theta(cx, theta)
        back.populate_tables()
        back.select_gx(gx, cx, **kwargs)
        print(r'[/back] reselected theta=%r' % theta)
        print('')

    @slot_()
    @blocking
    @profile
    def delete_chip(back):
        # Action -> Delete Chip
        # RCOS TODO: Are you sure?
        cx = back.get_selected_cx()
        if cx is None:
            back.user_info('Cannot delete chip. No chip selected')
            return
        gx = back.hs.cx2_gx(cx)
        back.hs.delete_chip(cx)
        back.populate_tables()
        back.select_gx(gx)
        print('[back] deleted cx=%r\n' % cx)
        print('')

    @slot_()
    @blocking
    @profile
    def delete_image(back, gx=None):
        if gx is None:
            gx = back.get_selected_gx()
        if gx is None:
            back.user_info('Cannot delete image. No image selected')
            return
        back.clear_selection()
        back.hs.delete_image(gx)
        back.populate_tables()
        print('[back] deleted gx=%r\n' % gx)
        print('')

    @slot_()
    @blocking
    @profile
    def select_next(back):
        # Action -> Next
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

    @slot_()
    @blocking
    def precompute_feats(back):
        # Batch -> Precompute Feats
        #prevBlock = back.front.blockSignals(True)
        back.hs.update_samples()
        back.hs.refresh_features()
        #back.front.blockSignals(prevBlock)
        back.populate_chip_table()
        print('')

    @slot_()
    @blocking
    def precompute_queries(back):
        # Batch -> Precompute Queries
        # TODO:
        #http://stackoverflow.com/questions/15637768/
        # pyqt-how-to-capture-output-of-pythons-interpreter-
        # and-display-it-in-qedittext
        #prevBlock = back.front.blockSignals(True)
        #import matching_functions as mf
        #import DataStructures as ds
        #import match_chips3 as mc3
        import sys
        back.precompute_feats()
        valid_cx = back.hs.get_valid_cxs()
        #if back.params.args.quiet:
            #mc3.print_off()
            #ds.print_off()
            #mf.print_off()
        fmtstr = util.progress_str(len(valid_cx), '[back*] Query qcx=%r: ')
        for count, qcx in enumerate(valid_cx):
            sys.stdout.write(fmtstr % (qcx, count))
            back.hs.query(qcx, dochecks=False)
            if count % 100 == 0:
                sys.stdout.write('\n ...')
        sys.stdout.write('\n ...')
        #mc3.print_on()
        #ds.print_on()
        #mf.print_on()
        print('')
        #back.front.blockSignals(prevBlock)

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------

    #@slot_(rundbg=True)
    @slot_()
    @blocking
    def layout_figures(back):
        # Options -> Layout Figures
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

    @slot_()
    def edit_preferences(back):
        # Options -> Edit Preferences
        back.edit_prefs = back.hs.prefs.createQWidget()
        epw = back.edit_prefs
        epw.ui.defaultPrefsBUT.clicked.connect(back.default_preferences)
        query_uid = ''.join(back.hs.prefs.query_cfg.get_uid())
        print('[*back] query_uid = %s' % query_uid)
        print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------

    @slot_()
    def view_docs(back):
        from hscom import cross_platform as cplat
        hsdir = io.get_hsdir()
        pdf_dpath = join(hsdir, '_doc')
        pdf_fpath = join(pdf_dpath, 'HotSpotterUserGuide.pdf')
        cplat.startfile(pdf_fpath)

    @slot_()
    def view_database_dir(back):
        # Help -> View Directory Slots
        back.hs.vdd()

    @slot_()
    def view_computed_dir(back):
        back.hs.vcd()

    @slot_()
    def view_global_dir(back):
        back.hs.vgd()

    @slot_()
    def delete_cache(back):
        # Help -> Delete Directory Slots
        # RCOS TODO: Are you sure?
        ans = back.user_option('Are you sure you want to delete cache?')
        if ans != 'Yes':
            return
        back.invalidate_result()
        df2.close_all_figures()
        back.hs.delete_cache()
        back.populate_result_table()

    @slot_()
    def delete_global_prefs(back):
        # RCOS TODO: Are you sure?
        df2.close_all_figures()
        back.hs.delete_global_prefs()

    @slot_()
    def delete_queryresults_dir(back):
        # RCOS TODO: Are you sure?
        df2.close_all_figures()
        back.invalidate_result()
        back.hs.delete_queryresults_dir()
        back.populate_result_table()

    def invalidate_result(back):
        back.current_res = None

    @slot_()
    @blocking
    def dev_mode(back):
        # Help -> Developer Help
        steal_again = back.front.return_stdout()
        hs = back.hs    # NOQA
        front = back.front
        wasBlocked = front.blockSignals(True)
        devmode = True  # NOQA
        #print(util.indent(str(hs), '[*back.hs] '))
        #rrr()
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
        execstr = util.ipython_execstr()
        #print(execstr)
        print('Debugging in IPython. IPython will break gui until you exit')
        exec(execstr)
        if steal_again:
            back.front.steal_stdout()
        back.front.blockSignals(wasBlocked)
        #back.timer.start()

    @slot_()
    @blocking
    def dev_reload(back):
        # Help -> Developer Reload
        _dev_reload(back)

    @slot_()
    @blocking
    def detect_dupimg(back):
        back.hs.dbg_duplicate_images()
