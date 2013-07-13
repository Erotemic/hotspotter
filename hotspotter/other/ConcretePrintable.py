from hotspotter.other.AbstractPrintable import AbstractPrintable
import numpy as np
from logger import logdbg, logerr, func_log, logmsg
from PyQt4.Qt import QAbstractItemModel, QModelIndex, QVariant, Qt, QObject, QComboBox
import traceback
import os.path
import sys
import cPickle
import types

class DynStruct(AbstractPrintable):
    ' dynamical add and remove members '
    def __init__(self, child_exclude_list=[], copy_dict=None, copy_class=None):
        super(DynStruct, self).__init__(child_exclude_list)
        if type(copy_dict) == types.DictType:
            self.add_dict(copy_dict)
        #if copy_class != None and isinstance(copy_class, object):
            #import inspect
            #self.copied_class_str = repr(copy_class)
           # self.add_dict({name:attribute for (name, attribute) in inspect.getmembers(copy_class) if name.find('__') != 0 and str(type(attribute)) != "<type 'builtin_function_or_method'>" and str(type(attribute)) != "<type 'instancemethod'>"})

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
        'Adds a dictionary to the prefs'
        if type(dyn_dict) != types.DictType:
            raise Exception\
                    ('DynStruct.add_dict expects a dictionary.'+\
                     'Recieved: '+str(type(dyn_dict)))
        for (key,val) in dyn_dict.iteritems():
            self[key] = val

    def to_dict(self):
        '''Converts dynstruct to a dictionary.  '''
        dyn_dict = {}
        for (key, val) in self.__dict__.iteritems():
            if not key in self._printable_exclude:
                dyn_dict[key] = val
        return dyn_dict

    def execstr(self, local_name):
        '''returns a string which when evaluated will
           add the stored variables to the current namespace
           
           localname is the name of the variable in the current scope
           * use locals().update(dyn.to_dict()) instead
        '''
        execstr = ''
        for (key, val) in self.__dict__.iteritems():
            if not key in self._printable_exclude:
                execstr+=key+' = '+local_name+'.'+key+'\n'
        return execstr
        #return '''
#for (key, val) in %s.__dict__.iteritems():
    #if not key in %s._printable_exclude:
        #exec(key+' = %s.'+key)''' % tuple([local_name]*3)
            
#---------------

class Pref(DynStruct):
    '''
    Structure for Creating Preferences. 
    Caveats: 
        When using a value call with ['valname'] to be safe
    Features: 
      * Can be saved and loaded.
      * Can be nested 
      * Dynamically add/remove
    '''
    def __init__(self,\
                 default=None,\
                 doc='',\
                 min=None,\
                 max=None,\
                 hidden=False,\
                 choices=None,\
                 depeq=None,\
                 fpath='',\
                 copy_dict=None,\
                 parent=None):
        '''Creates a pref struct that will save itself to pref_fpath if
        available and have initail members of some dictionary'''
        super(Pref, self).__init__(child_exclude_list=\
                                   ['_intern', '_tree'], copy_dict=copy_dict)
        # Define internal structures
        self._intern = DynStruct()
        self._tree = DynStruct()
        self._intern.name    = 'root'
        self._intern.value   = None
        self._intern.type    = Pref
        self._intern.hidden  = hidden
        self._intern.fpath   = fpath
        self._intern.depeq   = depeq

        self._tree.parent    = parent
        self._tree.hidden_children = []
        self._tree.child_list  = []
        self._tree.child_names = []
        self._tree.num_visible_children = 0
        self._tree.aschildx    = 0 
        # Check if this is a leaf node and if so the type
        if choices != None:
            sel = 0
            if type(default) == types.IntType:
                sel = default
            elif type(default) == types.StringType:
                sel = choices.index(default)
                if sel == -1:
                    raise Exception('Default: %r is not in the chioces: %r '\
                                    % (default, choices) )
            default = (sel, choices)
            self._intern.type = 'combo'
        elif default != None:
            self._intern.type = type(default)
        self._intern.value = default
    
    def get_printable(self, type_bit=True, print_exclude_aug=[]):
        # Remove unsatisfied dependencies from the printed structure
        further_aug = print_exclude_aug[:]
        for child_name in self._tree.child_names:
            depeq = self[child_name+'_internal']._intern.depeq
            if not depeq is None and depeq[0].value() != depeq[1]:
                further_aug.append(child_name)
        return super(Pref, self).get_printable(type_bit, print_exclude_aug=further_aug)

    def value(self):
        if self._intern.type == Pref:
            return self
        elif self._intern.type == 'combo':
            return self.combo_val()
        else:
            return self._intern.value #TODO AS REFERENCE

    def combo_val(self):
        (sel, choices) = self._intern.value
        return choices[sel]

    def full_name(self):
        if self._tree.parent == None:
            return self._intern.name
        return self._tree.parent.full_name()+'.'+self._intern.name

    def customPrintableType(self, name):
        if name in self._tree.child_names:
            row = self._tree.child_names.index(name)
            child = self._tree.child_list[row] # child node to "overwrite"
            _typestr = self._tree.child_list[row]._intern.type
            if isinstance(_typestr, str):
                return _typestr
        return None


    def change_combo_val(self, new_val):
        '''Checks to see if a selection is a valid index or choice of
        a combo preference'''
        (sel, choices) = self._intern.value
        if new_val in choices:
            new_sel = choices.index(new_val)
        elif type(new_val) == types.IntType and\
                new_val < len(choices) and\
                new_val >= 0:
            new_sel = new_val
        else:
            logerr('The available choices are: '+str(self._intern.value[1]))
        return (new_sel, choices)

    def __overwrite_attr(self, name, attr):
        logdbg( "Overwriting Attribute: %r %r" % (name, attr) )
        row = self._tree.child_names.index(name)
        child = self._tree.child_list[row] # child node to "overwrite"
        if type(attr) == Pref:
            if attr._intern.type == Pref:
                # Main Branch Logic
                for (key, val) in attr.iteritems():
                    child.__setattr__(key, val)
            else:
                self.__overwrite_attr(name, attr.value())
        else: # Main Leaf Logic: 
            assert child._intern.type != Pref, self.full_name()+' Must be a leaf'
            if child._intern.type == 'combo':
                newval = child.change_combo_val(attr)
            else:
                newval = attr
            # Keep user-readonly map up to date with internals
            child._intern.value = newval
            self.__dict__[name] = child.value() 

    def __new_attr(self, name, attr):
        ' --- New Attribute Wrapper ---'
        if type(attr) == Pref:
            logdbg( "New Attribute: %r %r" % (name, attr.value()) )
            # If The New Attribute already has a PrefWrapper
            new_childx = len(self._tree.child_names)
            attr._tree.parent = self     # Give Child Parent
            attr._intern.name = name     # Give Child Name
            if attr._intern.depeq == None:
                attr._intern.depeq = self._intern.depeq # Give Child Parent Dependencies
            if attr._intern.hidden:
                self._tree.hidden_children.append(new_childx)
                self._tree.hidden_children.sort()
            attr._intern.aschildx = new_childx # Used for QTIndexing
            # Add To Internal Tree Structure
            self._tree.child_names.append(name)
            self._tree.child_list.append(attr)
            self.__dict__[name] = attr.value() 
        else:
            # If no wrapper, create one and readd
            if attr == None:
                attr = 'None'
            pref_attr = Pref(default=attr)
            self.__new_attr(name, pref_attr)

    # Attributes are children
    def __setattr__(self, name, attr):
        'Wraps objects as preferences if not done already'
        if len(name) > 0 and name[0] == '_':
            # Default behavior for _printable_exclude, _intern, _tree, etc
            return super(DynStruct, self).__setattr__(name, attr)
        # --- Overwrite Attribute ---
        if name in self._tree.child_names:
            self.__overwrite_attr(name, attr)
        else:
            self.__new_attr(name, attr)

    def __getattr__(self, name):
        'called as last resort. Allows easy access to internal prefs'
        if len(name) > 9 and name[-9:] == '_internal':
            attrx = self._tree.child_names.index(name[:-9])
            return self._tree.child_list[attrx]
        raise AttributeError('attribute: %s not found' % name)

    def iteritems(self):
        for (key, val) in self.__dict__.iteritems():
            if key in self._printable_exclude: 
                continue
            yield (key, val)

    def to_dict(self, split_structs_bit=False):
        '''Converts prefeters to a dictionary. 
        Children Pref can be optionally separated'''
        pref_dict = {}
        struct_dict = {}
        for (key, val) in self.iteritems():
            if split_structs_bit and isinstance(val, Pref):
                struct_dict[key] = val
                continue
            pref_dict[key] = val
        if split_structs_bit:
            return (pref_dict, struct_dict)
        return pref_dict

    def save(self):
        'Saves prefeters to disk in the form of a dict'
        if self._intern.fpath in ['', None]: 
            if self._tree.parent != None:
                logdbg('Can my parent save me?')
                return self._tree.parent.save()
            logdbg('I cannot be saved. I have no parents.')
            return False
        with open(self._intern.fpath, 'w') as f:
            logdbg('Saving to '+self._intern.fpath)
            pref_dict = self.to_dict()
            cPickle.dump(pref_dict, f)
        return True

    def load(self):
        'Read pref dict stored on disk. Overwriting current values.'
        if not os.path.exists(self._intern.fpath):
            return False
        with open(self._intern.fpath, 'r') as f:
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
        if type(self[key]) != types.BooleanType:
        #if not self._intern.type != types.BooleanType:
            raise Exception('Cannot toggle the non-boolean type: '+str(key))
        self.update(key, not self[key])

    def __str__(self):
        if self._intern.value != None:
            ret = super(DynStruct, self).__str__().replace('\n    ','')
            ret += 'LEAF '+repr(self._intern.name)+':'+repr(self._intern.value)
            return ret
        else:
            return super(DynStruct, self).__str__()

    @func_log
    def update(self, key, new_val):
        'Changes a prefeters value and saves it to disk'
        logdbg('Updating Preference: %s to %r' % (key, str(new_val)))
        self.__setattr__(key, new_val)
        return self.save()

    # QT THINGS
    def createQPreferenceModel(self):
        'Creates a QStandardItemModel that you can connect to a QTreeView'
        return QPreferenceModel(self)

    def qt_get_parent(self):
        return self._tree.parent

    def qt_parents_index_of_me(self):
        return self._tree.aschildx

    def qt_get_child(self, row):
        row_offset = (np.array(self._tree.hidden_children) <= row).sum()
        return self._tree.child_list[row + row_offset]

    def qt_row_count(self):
        return len(self._tree.child_list) - len(self._tree.hidden_children)

    def qt_col_count(self):
        return 2

    def qt_get_data(self, column):
        if column == 0:
            return self._intern.name
        data = self.value()
        if type(data) == Pref: # Recursive Case: Pref
            data = ''
        return data

    def qt_is_editable(self):
        uneditable_hack = ['database_dpath', 'use_thumbnails', 'thumbnail_size', 'kpts_extractor']
        self._intern.depeq
        if self._intern.name in uneditable_hack:
            return False
        if self._intern.depeq != None:
            return self._intern.depeq[0].value() == self._intern.depeq[1]
        return self._intern.value != None

    @func_log
    def qt_set_leaf_data(self, qvar):
        'Sets backend data using QVariants'
        if self._tree.parent == None: 
            raise Exception('Cannot set root preference')
        if self.qt_is_editable():
            new_val = 'BadThingsHappenedInPref'
            if self._intern.type == Pref:
                raise Exception('Qt can only change leafs')
            elif self._intern.type == types.IntType:
                new_val = int(qvar.toInt()[0])
            elif self._intern.type == types.StringType:
                new_val = str(qvar.toString())
            elif self._intern.type == types.FloatType:
                new_val = float(qvar.toFloat()[0])
            elif self._intern.type == types.BooleanType:
                new_val = bool(qvar.toBool())
            elif self._intern.type == 'combo':
                new_val = qvar.toString()
             # Save to disk
            return self._tree.parent.update(self._intern.name, new_val)
        return 'PrefNotEditable'

# Decorator to help catch errors that QT wont report
def report_thread_error(fn):
    def report_thread_error_wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs);
            return ret
        except Exception as ex:
            logmsg('\n\n *!!* Thread Raised Exception: '+str(ex))
            logmsg('\n\n *!!* Thread Exception Traceback: \n\n'+traceback.format_exc())
            sys.stdout.flush()
            et, ei, tb = sys.exc_info()
            raise 
    return report_thread_error_wrapper

class QPreferenceModel(QAbstractItemModel):
    'Convention states only items with column index 0 can have children'
    @report_thread_error
    def __init__(self, pref_struct, parent=None):
        super(QPreferenceModel, self).__init__(parent)
        self.rootPref  = pref_struct
    #-----------
    @report_thread_error
    def index2Pref(self, index=QModelIndex()):
        '''Internal helper method'''
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootPref
    #-----------
    # Overloaded ItemModel Read Functions
    @report_thread_error
    def rowCount(self, parent=QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_row_count()

    @report_thread_error
    def columnCount(self, parent=QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_col_count()

    @report_thread_error
    def data(self, index, role=Qt.DisplayRole):
        '''Returns the data stored under the given role 
        for the item referred to by the index.'''
        if not index.isValid():
            return QVariant()
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariant()
        nodePref = self.index2Pref(index)
        return QVariant(nodePref.qt_get_data(index.column()))

    @report_thread_error
    def index(self, row, col, parent=QModelIndex()):
        '''Returns the index of the item in the model specified
        by the given row, column and parent index.'''
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()
        parentPref = self.index2Pref(parent)
        childPref  = parentPref.qt_get_child(row)
        if childPref:
            return self.createIndex(row, col, childPref)
        else:
            return QModelIndex()

    @report_thread_error
    def parent(self, index=None):
        '''Returns the parent of the model item with the given index.
        If the item has no parent, an invalid QModelIndex is returned.'''
        if index is None: # Overload with QObject.parent()
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        nodePref = self.index2Pref(index)
        parentPref = nodePref.qt_get_parent()
        if parentPref == self.rootPref:
            return QModelIndex()
        return self.createIndex(parentPref.qt_parents_index_of_me(), 0, parentPref)
    
    #-----------
    # Overloaded ItemModel Write Functions
    @report_thread_error
    def flags(self, index):
        'Returns the item flags for the given index.'
        if index.column() == 0:
            # The First Column is just a label and unchangable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag(0)
        childPref = self.index2Pref(index)
        if childPref:
            if childPref.qt_is_editable():
                return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemFlag(0)

    @report_thread_error
    def setData(self, index, data, role=Qt.EditRole):
        'Sets the role data for the item at index to value.'
        if role != Qt.EditRole:
            return False
        leafPref = self.index2Pref(index)
        result = leafPref.qt_set_leaf_data(data)
        if result == True:
            self.dataChanged.emit(index, index)
        return result

    @report_thread_error
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return QVariant("Pref Name")
            if section == 1:
                return QVariant("Pref Value")
        return QVariant()
