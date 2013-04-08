from other.AbstractPrintable import AbstractPrintable
import os.path
import cPickle
import types

class DynStruct(AbstractPrintable):
    ' dynamical add and remove members '
    def __init__(self, child_exclude_list=[]):
        super(DynStruct, self).__init__(child_exclude_list)

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
#---------------
class PrefStruct(DynStruct):
    '''Structure for Creating Preferences. 
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
        super(PrefStruct, self).__init__(child_exclude_list=['pref_fpath'])
        self.pref_fpath = pref_fpath
        if type(copy_dict) == types.DictType:
            self.add_dict(copy_dict)

    def __getitem__(self, key):
        val = super(PrefStruct, self).__getitem__(key)
        if isinstance(val, list):
            raise NotImplementedError
        if isinstance(val, PrefStruct.ComboPref):
            return val()
        return val

    class ComboPref(object):
        '''Used when there are discrete values preferences can take
        If a value can be another PrefStruct, which can hold more preferences
        '''
        def __init__(self, sel, vals):
            self.sel = sel
            self.vals = vals
        def __call__(self):
            return self.vals[self.sel]

    def iteritems(self):
        for (key, val) in self.__dict__.iteritems():
            if key in self._printable_exclude: 
                continue
            yield (key, val)

    def to_dict(self, split_children_bit=False):
        '''Converts preferences to a dictionary. 
        Children PrefStructs can be optionally separated'''
        pref_dict = {}
        child_dict = {}
        for (key, val) in self.iteritems():
            if split_children_bit and isinstance(val, PrefStruct):
                child_dict[key] = val
                continue
            pref_dict[key] = val
        if split_children_bit:
            return (pref_dict, child_dict)
        return pref_dict

    def add_dict(self, pref_dict):
        'Adds a dictionary to the preferences'
        if type(pref_dict) != types.DictType:
            raise Exception\
                    ('PrefStruct.add_dict expects a dictionary.'+\
                     'Recieved: '+str(type(pref_dict)))
        for (key,val) in pref_dict.iteritems():
            self[key] = val

    def save(self):
        'Saves preferences to disk in the form of a dict'
        with open(self.pref_fpath, 'w') as f:
            pref_dict = self.to_dict()
            #cPickle.dump(pref_dict, f)

    def load(self):
        'Read pref dict stored on disk. Overwriting current values.'
        return False
        if not os.path.exists(self.pref_fpath):
            return False
        with open(self.pref_fpath, 'r') as f:
            pref_dict = cPickle.load(f)
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
        new_data = item.data()
        print new_data
        print item

    def createQtItemModel(self):
        'Creates a QStandardItemModel that you can connect to a QTreeView'
        from PyQt4.Qt import QStandardItemModel
        pref_model = QStandardItemModel()
        pref_model.setHorizontalHeaderLabels(['Pref Key', 'Pref Val'])
        prev_block = pref_model.blockSignals(True)
        parentItem = pref_model.invisibleRootItem()
        # Recursive Helper Function
        def populate_model_helper(parent_item, pref_struct):
            'populates the setting table based on the type of data in a dict or PrefStruct'
            parent_item.setColumnCount(2)
            parent_item.setRowCount(len(some_dict))
            # Define QtItemFlags for various pref types
            disabledFlags = 0
            scalarFlags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
            booleanFlags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
            unknownFlags = Qt.ItemIsSelectable
            for row_x, (name, data) in enumerate(populate_model_helper.iteritems()):
                name_column = 0
                data_column = 1
                name_item = QStandardItem()
                name_item.setData(name, Qt.DisplayRole)
                name_item.setFlags(Qt.ItemIsEnabled);
                parent_item.setChild(row_x, name_column, name_item)
                if type(data) == PrefStruct: # Recursive Case: PrefStruct
                    data_item = QStandardItem()
                    data_item.setFlags(Qt.ItemFlags(disabledFlags)); 
                    parent_item.setChild(row_x, data_column, data_item)
                    populate_model_helper(name_item, data)
                    continue
                # Base Case: Column Item
                data_item = QStandardItem()
                if type(data) == types.IntType:
                    data_item.setData(data, Qt.DisplayRole)
                    data_item.setFlags(scalarFlags);
                elif type(data) == types.StringType:
                    data_item.setData(str(data), Qt.DisplayRole)
                    data_item.setFlags(scalarFlags);
                elif type(data) == types.BooleanType:
                    data_item.setCheckState([Qt.Unchecked, Qt.Checked][data])
                    data_item.setFlags(booleanFlags);
                elif type(data) == PrefStruct.ComboPref:
                    data_item.setData(repr(data), Qt.DisplayRole)
                    data_item.setFlags(unknownFlags);
                else:
                    data_item.setData(repr(data), Qt.DisplayRole)
                    data_item.setFlags(unknownFlags);
                # Add New Column Item to the tree
                parent_item.setChild(row_x, data_column, data_item)
            #end populate_helper
        populate_model_helper(parentItem, self)
        pref_model.blockSignals(prev_block)
        pref_model.itemChanged.connect(self.itemChangedSlot)
        return pref_model
        #epw.pref_skel.prefTreeView.setModel(epw.pref_model)
        #epw.pref_skel.prefTreeView.header().resizeSection(0,250)

