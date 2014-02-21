from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[front]', DEBUG=False)
# Python
import sys
# Qt
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import (QAbstractItemView, pyqtSignal, Qt)
# HotSpotter
from _frontend.MainSkel import Ui_mainSkel
import guitools
from guitools import slot_
from guitools import frontblocking as blocking
from hscom import tools

#=================
# Globals
#=================

IS_INIT = False
NOSTEAL_OVERRIDE = False  # Hard disable switch for stream stealer


#=================
# Decorators / Helpers
#=================

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

def clicked(func):
    def clicked_wrapper(front, item, *args, **kwargs):
        if front.isItemEditable(item):
            front.print('[front] does not select when clicking editable column')
            return
        if item == front.prev_tbl_item:
            return
        front.prev_tbl_item = item
        return func(front, item, *args, **kwargs)
    clicked_wrapper.func_name = func.func_name
    # Hacky decorator
    return clicked_wrapper


def csv_sanatize(str_):
    return str(str_).replace(',', ';;')


#=================
# Stream Stealer
#=================
import logging


class GUILoggingSender(QtCore.QObject):
    write_ = QtCore.pyqtSignal(str)

    def __init__(self, front):
        super(GUILoggingSender, self).__init__()
        self.write_.connect(front.gui_write)

    def write_gui(self, msg):
        self.write_.emit(str(msg))


class GUILoggingHandler(logging.StreamHandler):
    """
    A handler class which sends messages to to a connected QSlot
    """
    def __init__(self, front):
        super(GUILoggingHandler, self).__init__()
        self.sender = GUILoggingSender(front)

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            self.sender.write_.emit(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class StreamStealer(QtCore.QObject):
    write_ = QtCore.pyqtSignal(str)
    flush_ =  QtCore.pyqtSignal()

    def __init__(self, front, parent=None, share=False):
        super(StreamStealer, self).__init__(parent)
        # Define the Stream Stealer write function
        if share:
            self.write = self.write_shared
        else:
            self.write = self.write_gui
        self.write_.connect(front.gui_write)
        self.flush_.connect(front.gui_flush)
        # Do the stealing
        #stream_holder = sys
        #try:
            #__IPYTHON__
            #print('[front] detected __IPYTHON__')
            #from IPython.utils import io as iio
            #stream_holder = iio.IOTerm(None, self)
            #return
        #except NameError:
            #print('[front] did not detect __IPYTHON__')
            #pass
        #except Exception as ex:
            #print(ex)
            #raise

        # Remember which stream you've stolen
        self.iostream = sys.stdout
        self.iostream2 = sys.stderr
        # Redirect standard out to the StreamStealer object
        sys.stderr = self
        sys.stdout = self
        #steam_holder.stdout

    def write_shared(self, msg):
        msg_ = unicode(str(msg))
        self.iostream.write(msg_)
        self.write_.emit(msg_)

    def write_gui(self, msg):
        self.write_.emit(str(msg))

    def flush(self):
        self.flush_.emit()


def _steal_stdout(front):
    from hscom import params
    #front.ui.outputEdit.setPlainText(sys.stdout)
    nosteal = params.args.nosteal
    noshare = params.args.noshare
    if '--cmd' in sys.argv:
        nosteal = noshare = True
    #from IPython.utils import io
    #with io.capture_output() as captured:
        #%run my_script.py
    if NOSTEAL_OVERRIDE or (nosteal and noshare):
        print('[front] not stealing stdout.')
        return
    print('[front] stealing standard out')
    if front.ostream is None:
        # Connect a StreamStealer object to the GUI output window
        if '--nologging' in sys.argv:
            front.ostream = StreamStealer(front, share=not noshare)
        else:
            front.gui_logging_handler = GUILoggingHandler(front)
            __common__.add_logging_handler(front.gui_logging_handler)
    else:
        print('[front] stream already stolen')


def _return_stdout(front):
    #front.ui.outputEdit.setPlainText(sys.stdout)
    print('[front] returning standard out')
    if front.ostream is not None:
        sys.stdout = front.ostream.iostream
        sys.stderr = front.ostream.iostream2
        front.ostream = None
        return True
    else:
        print('[front] stream has not been stolen')
        return False


#=================
# Initialization
#=================


def init_ui(front):
    ui = Ui_mainSkel()
    ui.setupUi(front)
    return ui


def connect_file_signals(front):
    ui = front.ui
    back = front.back
    ui.actionNew_Database.triggered.connect(back.new_database)
    ui.actionOpen_Database.triggered.connect(back.open_database)
    ui.actionSave_Database.triggered.connect(back.save_database)
    ui.actionImport_Img_file.triggered.connect(back.import_images_from_file)
    ui.actionImport_Img_dir.triggered.connect(back.import_images_from_dir)
    ui.actionQuit.triggered.connect(back.quit)


def connect_action_signals(front):
    ui = front.ui
    back = front.back
    ui.actionAdd_Chip.triggered.connect(back.add_chip)
    ui.actionNew_Chip_Property.triggered.connect(back.new_prop)
    ui.actionQuery.triggered.connect(back.query)
    ui.actionReselect_Ori.triggered.connect(back.reselect_ori)
    ui.actionReselect_ROI.triggered.connect(back.reselect_roi)
    ui.actionDelete_Chip.triggered.connect(back.delete_chip)
    ui.actionDelete_Image.triggered.connect(back.delete_image)
    ui.actionNext.triggered.connect(back.select_next)


def connect_option_signals(front):
    ui = front.ui
    back = front.back
    ui.actionLayout_Figures.triggered.connect(back.layout_figures)
    ui.actionPreferences.triggered.connect(back.edit_preferences)
    #ui.actionTogPts.triggered.connect(back.toggle_points)


def connect_help_signals(front):
    ui = front.ui
    back = front.back
    msg_event = lambda title, msg: lambda: guitools.msgbox(title, msg)
    ui.actionView_Docs.triggered.connect(back.view_docs)
    ui.actionView_DBDir.triggered.connect(back.view_database_dir)
    ui.actionView_Computed_Dir.triggered.connect(back.view_computed_dir)
    ui.actionView_Global_Dir.triggered.connect(back.view_global_dir)

    ui.actionAbout.triggered.connect(msg_event('About', 'hotspotter'))
    ui.actionDelete_computed_directory.triggered.connect(back.delete_cache)
    ui.actionDelete_global_preferences.triggered.connect(back.delete_global_prefs)
    ui.actionDelete_Precomputed_Results.triggered.connect(back.delete_queryresults_dir)
    ui.actionDev_Mode_IPython.triggered.connect(back.dev_mode)
    ui.actionDeveloper_Reload.triggered.connect(back.dev_reload)
    #Taken care of by add_action ui.actionDetect_Duplicate_Images.triggered.connect(back.detect_dupimg)
    #ui.actionWriteLogs.triggered.connect(back.write_logs)


def connect_batch_signals(front):
    ui = front.ui
    back = front.back
    #ui.actionBatch_Change_Name.triggered.connect(back.batch_rename)
    ui.actionPrecomputeChipsFeatures.triggered.connect(back.precompute_feats)
    ui.actionPrecompute_Queries.triggered.connect(back.precompute_queries)
    #ui.actionScale_all_ROIS.triggered.connect(back.expand_rois)
    #ui.actionConvert_all_images_into_chips.triggered.connect(back.convert_images2chips)
    #ui.actionAddMetaProp.triggered.connect(back.add_chip_property)
    #ui.actionAutoassign.triggered.connect(back.autoassign)


def connect_experimental_signals(front):
    ui = front.ui
    back = front.back
    ui.actionMatching_Experiment.triggered.connect(back.actionRankErrorExpt)
    ui.actionName_Consistency_Experiment.triggered.connect(back.autoassign)


#def popup(front, pos):
    #for i in front.ui.gxs_TBL.selectionModel().selection().indexes():
        #front.print(repr((i.row(), i.column())))
    #menu = QtGui.QMenu()
    #action1 = menu.addAction("action1")
    #action2 = menu.addAction("action2")
    #action3 = menu.addAction("action2")
    #action = menu.exec_(front.ui.gxs_TBL.mapToGlobal(pos))
    #front.print('action = %r ' % action)

def new_menu_action(front, menu_name, name, text=None, shortcut=None, slot_fn=None):
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    ui = front.ui
    if hasattr(ui, action_name):
        raise Exception('menu action already defined')
    action = QtGui.QAction(front)
    setattr(ui, action_name, action)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    action.setObjectName(_fromUtf8(action_name))
    menu = getattr(ui, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    # TODO: Have ui.retranslate call this
    def retranslate_fn():
        printDBG('retranslating %s' % name)
        action.setText(QtGui.QApplication.translate("mainSkel", action_text, None, QtGui.QApplication.UnicodeUTF8))
        if action_shortcut is not None:
            action.setShortcut(QtGui.QApplication.translate("mainSkel", action_shortcut, None, QtGui.QApplication.UnicodeUTF8))
    def connect_fn():
        printDBG('connecting %s' % name)
        action.triggered.connect(slot_fn)
    connect_fn.func_name = name + '_' + connect_fn.func_name
    retranslate_fn.func_name = name + '_' + retranslate_fn.func_name
    front.connect_fns.append(connect_fn)
    front.retranslatable_fns.append(retranslate_fn)
    retranslate_fn()


def set_tabwidget_text(front, tblname, text):
    printDBG('[front] set_tabwidget_text(%s, %s)' % (tblname, text))
    tablename2_tabwidget = {
        'gxs': front.ui.image_view,
        'cxs': front.ui.chip_view,
        'nxs': front.ui.name_view,
        'res': front.ui.result_view,
    }
    ui = front.ui
    tab_widget = tablename2_tabwidget[tblname]
    tab_index = ui.tablesTabWidget.indexOf(tab_widget)
    tab_text = QtGui.QApplication.translate("mainSkel", text, None,
                                            QtGui.QApplication.UnicodeUTF8)
    ui.tablesTabWidget.setTabText(tab_index, tab_text)


class MainWindowFrontend(QtGui.QMainWindow):
    printSignal     = pyqtSignal(str)
    quitSignal      = pyqtSignal()
    selectGxSignal  = pyqtSignal(int)
    selectCidSignal = pyqtSignal(int)
    selectResSignal = pyqtSignal(int)
    selectNameSignal = pyqtSignal(str)
    changeCidSignal = pyqtSignal(int, str, str)
    aliasNameSignal = pyqtSignal(int, str, str)
    changeGxSignal  = pyqtSignal(int, str, bool)
    querySignal = pyqtSignal()

    def __init__(front, back):
        super(MainWindowFrontend, front).__init__()
        #print('[*front] creating frontend')
        front.prev_tbl_item = None
        front.ostream = None
        front.gui_logging_handler = None
        front.back = back
        front.ui = init_ui(front)
        # Programatially Defined Actions
        front.retranslatable_fns = []
        front.connect_fns = []
        new_menu_action(front, 'menuHelp', 'actionDetect_Duplicate_Images',
                        text='Detect Duplicate Images', slot_fn=back.detect_dupimg)
        # Progress bar is not hooked up yet
        front.ui.progressBar.setVisible(False)
        front.connect_signals()
        front.steal_stdout()

    def steal_stdout(front):
        return _steal_stdout(front)

    def return_stdout(front):
        return _return_stdout(front)

    # TODO: this code is duplicated in back
    def user_info(front, *args, **kwargs):
        return guitools.user_info(front, *args, **kwargs)

    def user_input(front, *args, **kwargs):
        return guitools.user_input(front, *args, **kwargs)

    def user_option(front, *args, **kwargs):
        return guitools.user_option(front, *args, **kwargs)

    @slot_()
    def closeEvent(front, event):
        #front.printSignal.emit('[*front] closeEvent')
        event.accept()
        front.quitSignal.emit()

    def connect_signals(front):
        # Connect signals to slots
        back = front.back
        ui = front.ui
        # Frontend Signals
        front.printSignal.connect(back.backend_print)
        front.quitSignal.connect(back.quit)
        front.selectGxSignal.connect(back.select_gx)
        front.selectCidSignal.connect(back.select_cid)
        front.selectResSignal.connect(back.select_res_cid)
        front.selectNameSignal.connect(back.select_name)
        front.changeCidSignal.connect(back.change_chip_property)
        front.aliasNameSignal.connect(back.alias_name)
        front.changeGxSignal.connect(back.change_image_property)
        front.querySignal.connect(back.query)

        # Menubar signals
        connect_file_signals(front)
        connect_action_signals(front)
        connect_option_signals(front)
        connect_batch_signals(front)
        #connect_experimental_signals(front)
        connect_help_signals(front)
        for func in front.connect_fns:
            func()
        #
        # Gui Components
        # Tables Widgets
        ui.cxs_TBL.itemClicked.connect(front.chip_tbl_clicked)
        ui.cxs_TBL.itemChanged.connect(front.chip_tbl_changed)
        ui.gxs_TBL.itemClicked.connect(front.img_tbl_clicked)
        ui.gxs_TBL.itemChanged.connect(front.img_tbl_changed)
        ui.res_TBL.itemClicked.connect(front.res_tbl_clicked)
        ui.res_TBL.itemChanged.connect(front.res_tbl_changed)
        ui.nxs_TBL.itemClicked.connect(front.name_tbl_clicked)
        ui.nxs_TBL.itemChanged.connect(front.name_tbl_changed)
        # Tab Widget
        ui.tablesTabWidget.currentChanged.connect(front.change_view)
        ui.cxs_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.res_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.gxs_TBL.sortByColumn(0, Qt.AscendingOrder)

    def print(front, msg):
        print('[*front*] ' + msg)
        #front.printSignal.emit('[*front] ' + msg)

    @slot_(bool)
    def setEnabled(front, flag):
        #front.printDBG('setEnabled(%r)' % flag)
        ui = front.ui
        # Enable or disable all actions
        for uikey in ui.__dict__.keys():
            if uikey.find('action') == 0:
                ui.__dict__[uikey].setEnabled(flag)

        # The following options are always enabled
        ui.actionOpen_Database.setEnabled(True)
        ui.actionNew_Database.setEnabled(True)
        ui.actionQuit.setEnabled(True)
        ui.actionAbout.setEnabled(True)
        ui.actionView_Docs.setEnabled(True)
        ui.actionDelete_global_preferences.setEnabled(True)

        # The following options are no implemented. Disable them
        ui.actionConvert_all_images_into_chips.setEnabled(False)
        ui.actionBatch_Change_Name.setEnabled(False)
        ui.actionScale_all_ROIS.setEnabled(False)
        ui.actionWriteLogs.setEnabled(False)
        ui.actionAbout.setEnabled(False)
        #ui.actionView_Docs.setEnabled(False)

    @slot_(str, list, list, list, list)
    @blocking
    def populate_tbl(front, tblname, col_fancyheaders, col_editable,
                     row_list, datatup_list):
        #front.printDBG('populate_tbl(%s)' % table_name)
        tblname = str(tblname)
        fancytab_dict = {
            'gxs': 'Image Table',
            'cxs': 'Chip Table',
            'nxs': 'Name Table',
            'res': 'Query Results Table',
        }
        tbl_dict = {
            'gxs': front.ui.gxs_TBL,
            'cxs': front.ui.cxs_TBL,
            'nxs': front.ui.nxs_TBL,
            'res': front.ui.res_TBL,
        }
        tbl = tbl_dict[tblname]
        #try:
            #tbl = front.ui.__dict__['%s_TBL' % tblname]
        #except KeyError:
            #ui_keys = front.ui.__dict__.keys()
            #tblname_list = [key for key in ui_keys if key.find('_TBL') >= 0]
            #msg = '\n'.join(['Invalid tblname = %s_TBL' % tblname,
                             #'valid names:\n  ' + '\n  '.join(tblname_list)])
            #raise Exception(msg)
        front._populate_table(tbl, col_fancyheaders, col_editable, row_list, datatup_list)
        # Set the tab text to show the number of items listed
        text = fancytab_dict[tblname] + ' : %d' % len(row_list)
        set_tabwidget_text(front, tblname, text)

    def _populate_table(front, tbl, col_fancyheaders, col_editable, row_list, datatup_list):
        # TODO: for chip table: delete metedata column
        # RCOS TODO:
        # I have a small right-click context menu working
        # Maybe one of you can put some useful functions in these?
        # RCOS TODO: How do we get the clicked item on a right click?
        # RCOS TODO:
        # The data tables should not use the item model
        # Instead they should use the more efficient and powerful
        # QAbstractItemModel / QAbstractTreeModel
        def set_header_context_menu(hheader):
            hheader.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('header', lambda: print('finishme')),
                ('cancel', lambda: print('cancel')), ]
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            hheader.customContextMenuRequested.connect(popup_slot)

        def set_table_context_menu(tbl):
            tbl.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('Query', front.querySignal.emit), ]
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            tbl.customContextMenuRequested.connect(popup_slot)

        hheader = tbl.horizontalHeader()
        #set_header_context_menu(hheader)
        #set_table_context_menu(tbl)

        sort_col = hheader.sortIndicatorSection()
        sort_ord = hheader.sortIndicatorOrder()
        tbl.sortByColumn(0, Qt.AscendingOrder)  # Basic Sorting
        tblWasBlocked = tbl.blockSignals(True)
        tbl.clear()
        tbl.setColumnCount(len(col_fancyheaders))
        tbl.setRowCount(len(row_list))
        tbl.verticalHeader().hide()
        tbl.setHorizontalHeaderLabels(col_fancyheaders)
        tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl.setSortingEnabled(False)
        #dbg_col2_dtype = {}
        #def DEBUG_COL_DTYPE(col, dtype):
            #if not dtype in dbg_col2_dtype:
                #dbg_col2_dtype[dtype] = [col]
            #else:
                #if not col in dbg_col2_dtype[dtype]:
                    #dbg_col2_dtype[dtype].append(col)
        # Add items for each row and column
        for row in iter(row_list):
            data_tup = datatup_list[row]
            for col, data in enumerate(data_tup):
                item = QtGui.QTableWidgetItem()
                # RCOS TODO: Pass in datatype here.
                # BOOLEAN DATA
                if tools.is_bool(data) or data == 'True' or data == 'False':
                    check_state = Qt.Checked if bool(data) else Qt.Unchecked
                    item.setCheckState(check_state)
                    #DEBUG_COL_DTYPE(col, 'bool')
                    #item.setData(Qt.DisplayRole, bool(data))
                # INTEGER DATA
                elif tools.is_int(data):
                    item.setData(Qt.DisplayRole, int(data))
                    #DEBUG_COL_DTYPE(col, 'int')
                # FLOAT DATA
                elif tools.is_float(data):
                    item.setData(Qt.DisplayRole, float(data))
                    #DEBUG_COL_DTYPE(col, 'float')
                # STRING DATA
                else:
                    item.setText(str(data))
                    #DEBUG_COL_DTYPE(col, 'string')
                # Mark as editable or not
                if col_editable[col]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    item.setBackground(QtGui.QColor(250, 240, 240))
                else:
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignHCenter)
                tbl.setItem(row, col, item)

        #print(dbg_col2_dtype)
        tbl.setSortingEnabled(True)
        tbl.sortByColumn(sort_col, sort_ord)  # Move back to old sorting
        tbl.show()
        tbl.blockSignals(tblWasBlocked)

    def isItemEditable(self, item):
        return int(Qt.ItemIsEditable & item.flags()) == int(Qt.ItemIsEditable)

    #=======================
    # General Table Getters
    #=======================

    def get_tbl_header(front, tbl, col):
        # Map the fancy header back to the internal one.
        fancy_header = str(tbl.horizontalHeaderItem(col).text())
        header = (front.back.reverse_fancy[fancy_header]
                  if fancy_header in front.back.reverse_fancy else fancy_header)
        return header

    def get_tbl_int(front, tbl, row, col):
        return int(tbl.item(row, col).text())

    def get_tbl_str(front, tbl, row, col):
        return str(tbl.item(row, col).text())

    def get_header_val(front, tbl, header, row):
        # RCOS TODO: This is hacky. These just need to be
        # in dicts to begin with.
        tblname = str(tbl.objectName()).replace('_TBL', '')
        tblname = tblname.replace('image', 'img')  # Sooooo hack
        # TODO: backmap from fancy headers to consise
        col = front.back.table_headers[tblname].index(header)
        return tbl.item(row, col).text()

    #=======================
    # Specific Item Getters
    #=======================

    def get_chiptbl_header(front, col):
        return front.get_tbl_header(front.ui.cxs_TBL, col)

    def get_imgtbl_header(front, col):
        return front.get_tbl_header(front.ui.gxs_TBL, col)

    def get_restbl_header(front, col):
        return front.get_tbl_header(front.ui.res_TBL, col)

    def get_nametbl_header(front, col):
        return front.get_tbl_header(front.ui.nxs_TBL, col)

    def get_restbl_cid(front, row):
        return int(front.get_header_val(front.ui.res_TBL, 'cid', row))

    def get_chiptbl_cid(front, row):
        return int(front.get_header_val(front.ui.cxs_TBL, 'cid', row))

    def get_nametbl_name(front, row):
        return str(front.get_header_val(front.ui.nxs_TBL, 'name', row))

    def get_nametbl_nx(front, row):
        return int(front.get_header_val(front.ui.nxs_TBL, 'nx', row))

    def get_imgtbl_gx(front, row):
        return int(front.get_header_val(front.ui.gxs_TBL, 'gx', row))

    #=======================
    # Table Changed Functions
    #=======================

    @slot_(QtGui.QTableWidgetItem)
    def img_tbl_changed(front, item):
        front.print('img_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_gx = front.get_imgtbl_gx(row)
        header_lbl = front.get_imgtbl_header(col)
        new_val = item.checkState() == Qt.Checked
        front.changeGxSignal.emit(sel_gx, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def chip_tbl_changed(front, item):
        front.print('chip_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_cid = front.get_chiptbl_cid(row)  # Get selected chipid
        new_val = csv_sanatize(item.text())   # sanatize for csv
        header_lbl = front.get_chiptbl_header(col)  # Get changed column
        front.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def res_tbl_changed(front, item):
        front.print('res_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_cid  = front.get_restbl_cid(row)  # The changed row's chip id
        new_val  = csv_sanatize(item.text())  # sanatize val for csv
        header_lbl = front.get_restbl_header(col)  # Get changed column
        front.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def name_tbl_changed(front, item):
        front.print('name_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_nx = front.get_nametbl_nx(row)    # The changed row's name index
        new_val  = csv_sanatize(item.text())  # sanatize val for csv
        header_lbl = front.get_nametbl_header(col)  # Get changed column
        front.aliasNameSignal.emit(sel_nx, header_lbl, new_val)

    #=======================
    # Table Clicked Functions
    #=======================
    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def img_tbl_clicked(front, item):
        row = item.row()
        front.print('img_tbl_clicked(%r)' % (row))
        sel_gx = front.get_imgtbl_gx(row)
        front.selectGxSignal.emit(sel_gx)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def chip_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('chip_tbl_clicked(%r, %r)' % (row, col))
        sel_cid = front.get_chiptbl_cid(row)
        front.selectCidSignal.emit(sel_cid)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def res_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('res_tbl_clicked(%r, %r)' % (row, col))
        sel_cid = front.get_restbl_cid(row)
        front.selectResSignal.emit(sel_cid)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def name_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('name_tbl_clicked(%r, %r)' % (row, col))
        sel_name = front.get_nametbl_name(row)
        front.selectNameSignal.emit(sel_name)

    #=======================
    # Other
    #=======================

    @slot_(int)
    def change_view(front, new_state):
        tab_name = str(front.ui.tablesTabWidget.tabText(new_state))
        front.print('change_view(%r)' % new_state)
        prevBlock = front.ui.tablesTabWidget.blockSignals(True)
        front.ui.tablesTabWidget.blockSignals(prevBlock)
        if tab_name.startswith('Query Results Table'):
            print(front.back.hs.get_cache_uid())

    @slot_(str, str, list)
    def modal_useroption(front, msg, title, options):
        pass

    @slot_(str)
    def gui_write(front, msg_):
        app = front.back.app
        outputEdit = front.ui.outputEdit
        # Write msg to text area
        outputEdit.moveCursor(QtGui.QTextCursor.End)
        # TODO: Find out how to do backspaces in textEdit
        msg = str(msg_)
        if msg.find('\b') != -1:
            msg = msg.replace('\b', '') + '\n'
        outputEdit.insertPlainText(msg)
        if app is not None:
            app.processEvents()

    @slot_()
    def gui_flush(front):
        app = front.back.app
        if app is not None:
            app.processEvents()
        #front.ui.outputEdit.moveCursor(QtGui.QTextCursor.End)
        #front.ui.outputEdit.insertPlainText(msg)
