from other.AbstractPrintable import AbstractPrintable
from logger import logdbg, logerr, func_log
from PyQt4.Qt import QAbstractItemModel, QModelIndex, QVariant, Qt, QObject, QComboBox
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
    def __init__(self, pref_fpath=None, copy_dict=None, parent=None):
        '''Creates a pref struct that will save itself to pref_fpath if
        available and have initail members of some dictionary'''
        super(PrefStruct, self).__init__(child_exclude_list=['pref_fpath', 'parent'], copy_dict=copy_dict)
        self.pref_fpath = pref_fpath
        self.parent = None
        if parent != None:
            self.setParent(parent)

    def setParent(self, parent):
        if type(parent) != PrefStruct:
            raise Exception('The parent of a PrefStruct must be a PrefStruct')
        self.parent = parent

    #def __getitem__(self, key):
        #val = super(PrefStruct, self).__getitem__(key)
        #if isinstance(val, list):
            #raise NotImplementedError
        #if isinstance(val, ComboPref):
            #return val()
        #return val

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

    @func_log
    def save(self):
        'Saves preferences to disk in the form of a dict'
        if self.pref_fpath is None: 
            if self.parent != None:
                logdbg('Can my parent save me?')
                return self.parent.save()
            logdbg('I cannot be saved. I have no parents.')
            return False
        with open(self.pref_fpath, 'w') as f:
            logdbg('Saving to '+self.pref_fpath)
            pref_dict = self.to_dict()
            cPickle.dump(pref_dict, f)
        return True

    @func_log
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
        'Toggles a boolean key'
        if not self[key] in [True, False]:
            raise Exception('Cannot toggle the non-boolean type: '+str(key))
        self.update(key, not self[key])

    @func_log
    def update(self, key, new_val):
        'Changes a preference value and saves it to disk'
        logdbg('Updating Preference: %s to %r' % (key, str(new_val)))
        self[key] = new_val
        return self.save()
       

    def createQPreferenceModel(self):
        'Creates a QStandardItemModel that you can connect to a QTreeView'
        return QPreferenceModel(self)


class StaticPrefTreeItem(object):
    '''This class represents one row in the Tree and builds itself from a PrefStruct'''
    def __init__(self, pref_name, pref_value, parent=None):
        self.parentItem = parent
        self.pref_name  = pref_name
        self.pref_value = pref_value #Changing this wont matter, it just cares about initial value
        self.childItems = []
        self.static_build_tree()

    def static_build_tree(self):
        'Add all children of any PrefStructs'
        self.childItems = []
        if type(self.pref_value) == PrefStruct:
            for (key, val) in self.pref_value.iteritems():
                self.childItems.append(StaticPrefTreeItem(key, val, self))

    def data(self, column):
        if column == 0:
            return self.pref_name
        assert column == 1, 'Cant have more than 2 columns right now'
        if type(self.pref_value) == PrefStruct: # Recursive Case: PrefStruct
            #return ' ---- #children='+str(len(self.childItems))
            return ''
        data = self.getPrefStructValue()
        ## Base Case: Column Item
        if type(data) in [types.IntType, types.StringType, types.FloatType]:
            return data
        elif type(data) == types.BooleanType:
            return data
        elif type(data) == ComboPref:
            return data() #Calling data gives you the string value of ComboPref

    def childNumber(self):
        if self.parentItem != None:
            return self.parentItem.childItems.index(self)
        return 0

    def isEditable(self):
        uneditable_hack = ['database_dpath', 'use_thumbnails', 'thumbnail_size', 'kpts_extractor', 'quantizer', 'indexer']
        if self.pref_name in uneditable_hack:
            return False
        return self.parentItem != None and type(self.pref_value) != PrefStruct

    def getPrefStructValue(self):
        'access the actual backend data'
        if self.parentItem != None:
            # Your parent better be a PrefStruct
            parent_prefs = self.parentItem.pref_value
            return parent_prefs[self.pref_name]

    @func_log
    def setPrefStructValue(self, qvar):
        'sets the actual backend data'
        if self.parentItem == None: 
            raise Exception('Cannot set root preference')
        parent_prefs = self.parentItem.pref_value
        if self.isEditable():
            old_val = parent_prefs[self.pref_name]
            new_val = 'BadThingsHappenedInPrefStructValue'
            logdbg('Editing PrefName=%r with OldValue=%r' % (self.pref_name, old_val))
            if type(old_val) == types.IntType:
                new_val = int(qvar.toInt()[0])
            elif type(old_val) == types.StringType:
                new_val = str(qvar.toString())
            elif type(old_val) == types.FloatType:
                new_val = float(qvar.toFloat()[0])
            elif type(old_val) == types.BooleanType:
                new_val = bool(qvar.toBool())
            elif type(old_val) == ComboPref:
                # Only allow the new string to be one of the combopref options
                logdbg('And it is a Combo Pref %r, %r' % (old_val.sel, old_val.vals))
                try: 
                    combo_sel = str(qvar.toString())
                    new_val     = old_val
                    new_index   = new_val.vals.index(combo_sel)
                    new_val.sel = new_index
                    logdbg('With new members: %r, %r' % (new_val.sel, new_val.vals))
                except: # Tell the user if she is wrong
                    logerr('Valid values for this pref: '+str(old_val.vals))
            logdbg('Changing to NewValue=%r' % new_val)
            return parent_prefs.update(self.pref_name, new_val) # Save to disk
        return 'PrefNotEditable'

    def setData(self, new_data):
        self.setPrefStructValue(new_data)
        self.pref_value = new_data


class QPreferenceModel(QAbstractItemModel):
    'Convention states only items with column index 0 can have children'
    def __init__(self, pref_struct, parent=None):
        super(QPreferenceModel, self).__init__(parent)
        self.rootItem  = StaticPrefTreeItem('staticRoot', pref_struct)
    #-----------
    def getItem(self, index=QModelIndex()):
        '''Internal helper method'''
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootItem
    #-----------
    # Overloaded ItemModel Read Functions
    def rowCount(self, parent=QModelIndex()):
        parentItem = self.getItem(parent)
        return len(parentItem.childItems)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def data(self, index, role=Qt.DisplayRole):
        '''Returns the data stored under the given role 
        for the item referred to by the index.'''
        if not index.isValid():
            return QVariant()
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariant()
        item = self.getItem(index)
        return QVariant(item.data(index.column()))

    def index(self, row, col, parent=QModelIndex()):
        '''Returns the index of the item in the model specified
        by the given row, column and parent index.'''
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()
        parentItem = self.getItem(parent)
        childItem  = parentItem.childItems[row]
        if childItem:
            return self.createIndex(row, col, childItem)
        else:
            return QModelIndex()

    def parent(self, index=None):
        '''Returns the parent of the model item with the given index.
        If the item has no parent, an invalid QModelIndex is returned.'''
        if index is None: # Overload with QObject.parent()
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        childItem = self.getItem(index)
        parentItem = childItem.parentItem
        if parentItem == self.rootItem:
            return QModelIndex()
        return self.createIndex(parentItem.childNumber(), 0, parentItem)
    
    #-----------
    # Overloaded ItemModel Write Functions
    def flags(self, index):
        'Returns the item flags for the given index.'
        if index.column() == 0:
            # The First Column is just a label and unchangable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag(0)
        childItem = self.getItem(index)
        if childItem:
            if childItem.isEditable():
                return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemFlag(0)

    def setData(self, index, value, role=Qt.EditRole):
        'Sets the role data for the item at index to value.'
        if role != Qt.EditRole:
            return False
        item = self.getItem(index)
        result = item.setData(value)
        if result:
            self.dataChanged.emit(index, index)
        return result

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return QVariant("Pref Name")
            if section == 1:
                return QVariant("Pref Value")
        return QVariant()
