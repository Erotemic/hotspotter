from other.AbstractPrintable import AbstractPrintable
import os.path
import cPickle
import types

class DynStruct(AbstractPrintable):
    ' dynamical add and remove members '
    def __init__(self, child_exclude_list=[], copy_dict=None, copy_class=None):
        super(DynStruct, self).__init__(child_exclude_list)
        if type(copy_dict) == types.DictType:
            self.add_dict(copy_dict)
        if copy_class != None and isinstance(copy_class, object):
            import inspect
            self.copied_class_str = repr(copy_class)
            self.add_dict({name:attribute for (name, attribute) in inspect.getmembers(copy_class) if name.find('__') != 0 and str(type(attribute)) != "<type 'builtin_function_or_method'>" and str(type(attribute)) != "<type 'instancemethod'>"})

    def dynget(self, *prop_list):
        return tuple([self.__dict__[prop_name] for prop_name in prop_list])
    def dynset(self, *propval_list):
        offset = len(propval_list)/2
        for i in range(offset):
            self.__dict__[propval_list[i]] = propval_list[i+offset]
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            for k, v in zip(key, value):
                setattr(self, k, v)
        else:
            setattr(self, key, value)
    def __getitem__(self, key):
        if isinstance(key, tuple):
            val = []
            for k in key:
                val.append(getattr(self, k))
        else:
            val = getattr(self, key)
        return val

    def add_dict(self, dyn_dict):
        'Adds a dictionary to the preferences'
        if type(dyn_dict) != types.DictType:
            raise Exception\
                    ('DynStruct.add_dict expects a dictionary.'+\
                     'Recieved: '+str(type(dyn_dict)))
        for (key,val) in dyn_dict.iteritems():
            self[key] = val
#---------------

class ComboPref(object):
    '''Used when there are discrete values preferences can take
    If a value can be another PrefStruct, which can hold more preferences
    '''
    def __init__(self, sel, vals):
        self.sel = sel
        self.vals = vals
    def __call__(self):
        return self.vals[self.sel]
    def __str__(self):
        return str(self.vals[self.sel])

class PrefStruct(DynStruct):
    '''
    Structure for Creating Preferences. 
    Caveats: 
        When using a value call with ['valname'] to be safe
    Features: 
      * Can be saved and loaded.
      * Can be nested 
      * Dynamically add/remove
    '''
    def __init__(self, pref_fpath=None, copy_dict=None):
        '''Creates a pref struct that will save itself to pref_fpath if
        available and have initail members of some dictionary'''
        super(PrefStruct, self).__init__(child_exclude_list=['pref_fpath'], copy_dict=copy_dict)
        self.pref_fpath = pref_fpath

    def __getitem__(self, key):
        val = super(PrefStruct, self).__getitem__(key)
        if isinstance(val, list):
            raise NotImplementedError
        if isinstance(val, ComboPref):
            return val()
        return val

    def iteritems(self):
        for (key, val) in self.__dict__.iteritems():
            if key in self._printable_exclude: 
                continue
            yield (key, val)

    def num_items(self):
        count = -1
        for count, item in enumerate(self.iteritems()):
            pass
        return count+1

    def to_dict(self, split_structs_bit=False):
        '''Converts preferences to a dictionary. 
        Children PrefStructs can be optionally separated'''
        pref_dict = {}
        struct_dict = {}
        for (key, val) in self.iteritems():
            if split_structs_bit and isinstance(val, PrefStruct):
                struct_dict[key] = val
                continue
            pref_dict[key] = val
        if split_structs_bit:
            return (pref_dict, struct_dict)
        return pref_dict


    def save(self):
        'Saves preferences to disk in the form of a dict'
        if self.pref_fpath is None: 
            return False
        with open(self.pref_fpath, 'w') as f:
            pref_dict = self.to_dict()
            cPickle.dump(pref_dict, f)
        return True

    def load(self):
        'Read pref dict stored on disk. Overwriting current values.'
        if not os.path.exists(self.pref_fpath):
            return False
        with open(self.pref_fpath, 'r') as f:
            try:
                pref_dict = cPickle.load(f)
            except EOFError:
                import warnings
                warnings.warn('Preference file did not load correctly')
                return False
        if type(pref_dict) != types.DictType:
            raise Exception('Preference file is corrupted')
        self.add_dict(pref_dict)
        return True

    def toggle(self, key):
        if not self[key] in [True, False]:
            raise Exception('Cannot toggle the non-boolean type: '+str(key))
        self[key] = not self[key]
        self.save()

    def update(self, key, val):
        self[key] = val
        self.save()

    def modelItemChangedSlot(self, item):
        from logger import logdbg, logmsg, logerr
        new_data = item.data().toPyObject()
        self.item = item
        if not hasattr(self,'item_list'):
            self.item_list = []
        self.item_list.append(item)

    def createQPreferenceModel(self):
        'Creates a QStandardItemModel that you can connect to a QTreeView'
        return QPreferenceModel(self)
        #from PyQt4.Qt import QStandardItemModel, Qt, QStandardItem
        #pref_model = QStandardItemModel()
        #pref_model.setHorizontalHeaderLabels(['Pref Key', 'Pref Val'])
        #prev_block = pref_model.blockSignals(True)
        #parentItem = pref_model.invisibleRootItem()
        ## Recursive Helper Function
        #def populate_model_helper(parent_item, pref_struct):
            #'populates the setting table based on the type of data in a dict or PrefStruct'
            #parent_item.setColumnCount(2)
            #parent_item.setRowCount(pref_struct.num_items())
            ## Define QtItemFlags for various pref types
            #disabledFlags = 0
            #scalarFlags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
            #booleanFlags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
            #unknownFlags = Qt.ItemIsSelectable
            #for row_x, (name, data) in enumerate(pref_struct.iteritems()):
                #name_column = 0
                #data_column = 1
                #name_item = QStandardItem()
                #name_item.setData(name, Qt.DisplayRole)
                #name_item.setFlags(Qt.ItemIsEnabled);
                #parent_item.setChild(row_x, name_column, name_item)
                #if type(data) == PrefStruct: # Recursive Case: PrefStruct
                    #data_item = QStandardItem()
                    #data_item.setFlags(Qt.ItemFlags(disabledFlags)); 
                    #parent_item.setChild(row_x, data_column, data_item)
                    #populate_model_helper(name_item, data)
                    #continue
                ## Base Case: Column Item
                #data_item = QStandardItem()
                #if type(data) == types.IntType:
                    #data_item.setData(data, Qt.DisplayRole)
                    #data_item.setFlags(scalarFlags);
                #elif type(data) == types.StringType:
                    #data_item.setData(str(data), Qt.DisplayRole)
                    #data_item.setFlags(scalarFlags);
                #elif type(data) == types.BooleanType:
                    #data_item.setCheckState([Qt.Unchecked, Qt.Checked][data])
                    #data_item.setFlags(booleanFlags);
                #elif type(data) == ComboPref:
                    #data_item.setData(repr(data), Qt.DisplayRole)
                    #data_item.setFlags(unknownFlags);
                #else:
                    #data_item.setData(repr(data), Qt.DisplayRole)
                    #data_item.setFlags(unknownFlags);
                ## Add New Column Item to the tree
                #parent_item.setChild(row_x, data_column, data_item)
            ##end populate_helper
        #populate_model_helper(parentItem, self)
        #pref_model.blockSignals(prev_block)
        #pref_model.itemChanged.connect(self.modelItemChangedSlot)
        #return pref_model
        ##epw.pref_skel.prefTreeView.setModel(epw.pref_model)
        ##epw.pref_skel.prefTreeView.header().resizeSection(0,250)

class StaticPrefStructWrapper(object):
    '''Used as basis for the QPrefItemModel'''
    def __init__(self, pref_struct, parent=None):
        self.parent = parent
        self.wrapped = pref_struct
        self.pref_names = []
        self.pref_vals  = []
        self.struct_names = []
        self.struct_vals  = [] 
        self.nPrefs = 0
        self.nStructs = 0
        self.nChildren = 0
        self.extract_static_prefs()

    def rc2_data(self, row, col):
        if col == 0:
            if row < self.nPrefs:
                return self.pref_names[row]
            elif row < self.nChildren:
                return self.struct_names[row-self.nPrefs]
        elif col == 1:
            if row < self.nPrefs:
                return self.pref_vals[row]
            elif row < self.nChildren:
                return None
        return None

    def rc2_child(self, row, col):
        # Index into the static table
        if col == 0:
            if row < self.nChildren and row >=self.nPrefs:
                return self.struct_vals[row-self.nPrefs]
        return None

    def childNumber(self):
        if self.parent is None:
            return 0
        return self.parent.struct_vals.index(self)+self.parent.nPrefs

    def extract_static_prefs(self):
        'Extract a static representation.'
        # Parse Children into nonrecursive and recurisve members
        prefs, structs = self.wrapped.to_dict(split_structs_bit=True)
        # Update static lists
        self.pref_names = prefs.keys()
        self.pref_vals  = prefs.values()
        self.struct_names = structs.keys()
        self.struct_vals  = [StaticPrefStructWrapper(struct, self) for struct in structs.itervalues()] 
        # Update static bookeeping
        self.nPrefs = len(self.pref_names)
        self.nStructs = len(self.struct_names)
        self.nChildren = self.nPrefs + self.nStructs

from logger import logdbg, logmsg, func_log
from PyQt4.Qt import QAbstractItemModel, QModelIndex, QVariant, Qt, QObject
class QPreferenceModel(QAbstractItemModel):
    def __init__(self, pref_struct, parent=None):
        super(QPreferenceModel, self).__init__(parent)
        self.staticRoot  = StaticPrefStructWrapper(pref_struct)

    #-----------
    # Non-Overloaded ItemModel Helper Functions
    def getStaticWrapper(self, index=QModelIndex()):
        #print DynStruct(copy_class=index)
        if index.isValid():
            staticItem = index.internalPointer()
            if staticItem:
                return staticItem
        return self.staticRoot
    
    #-----------
    # Overloaded ItemModel Read Functions
    def rowCount(self, parent=QModelIndex()):
        #logmsg(' IN ROW COUNT !!!! \n ******************')
        #logmsg( DynStruct(copy_class=parent) )
        staticParent = self.getStaticWrapper(parent)
        #logmsg( str(DynStruct(copy_class=staticParent) ))
        #logmsg(' RETURNING \n &&&&&&&&&&&&&&&&&&&&&&')
        return staticParent.nChildren

    def columnCount(self, parent=QModelIndex()):
        #logmsg( DynStruct(copy_class=parent) )
        return 2

    def data(self, index, role):
        #logmsg( DynStruct(copy_class=parent) )
        if not index.isValid():
            return QVariant()
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariant()
        staticItem = self.getStaticWrapper(index)
        return QVariant(staticItem.rc2_data(index.row(), index.column()))

    def index(self, row, col, parent=QModelIndex()):
        #logmsg( 'row=%r col=%r ' % (row, col))
        #logmsg( DynStruct(copy_class=parent) )
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()
        staticParent = self.getStaticWrapper(parent)
        staticChild  = staticParent.rc2_child(row, col)
        if staticChild:
            return self.createIndex(row, col, staticChild)
        else:
            return QModelIndex()

    def parent(self, index=None):
        #logmsg( DynStruct(copy_class=index) )
        if index is None: # Overload with QObject.parent()
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        staticChild = self.getStaticWrapper(index)
        staticParent = staticChild.parent
        if staticParent == self.staticRoot:
            return QModelIndex()
        return self.createIndex(staticParent.childNumber(), 0, staticParent)
    
    #-----------
    # Overloaded ItemModel Write Functions
    #@func_log
    #def flags(self, index):
        #if not index.isValid():
            #return 0
        #return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    #@func_log
    #def setData(self, index, value, role=Qt.EditRole):
        #if role != Qt.EditRole:
            #return False
        #item = self.getItem(index)
        #result = item.setData(index.column(), value)
        #if result:
            #self.dataChanged.emit(index, index)
            ##self.emit(QtCore.SIGNAL('dataChanged(const QModelIndex&, const QModelIndex&)'), index, index)
        #return result
