import numpy as np
import re
import types
import numpy as np
import os.path

def reload_module():
    import imp, sys
    print('[printable] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

class AbstractPrintable(object):
    'A base class that prints its attributes instead of the memory address'

    def __init__(self, child_print_exclude=[]):
        self._printable_exclude = ['_printable_exclude'] + child_print_exclude

    def __str__(self):
        head = printableType(self)
        body = self.get_printable(type_bit=True)
        body = re.sub('\n *\n *\n','\n\n',body)
        return head+('\n'+body).replace('\n','\n    ')

    def printme(self):
        print(str(self))

    def printme2(self, 
                 type_bit=True, 
                 print_exclude_aug = [],
                 val_bit=True, 
                 max_valstr=1000,
                 justlength=True):
        to_print = self.get_printable(type_bit=type_bit,
                                      print_exclude_aug=print_exclude_aug,
                                      val_bit=val_bit,
                                      max_valstr=max_valstr,
                                      justlength=justlength)
        print(to_print)

    def get_printable(self,
                      type_bit=True,
                      print_exclude_aug = [], 
                      val_bit=True,
                      max_valstr=1000,
                      justlength=False):
        body = ''
        attri_list = []
        exclude_key_list = list(self._printable_exclude)+list(print_exclude_aug)
        for (key, val) in self.__dict__.iteritems():
            if key in exclude_key_list: continue
            namestr = str(key)
            typestr = printableType(val, name=key, parent=self)
            if not val_bit:
                attri_list.append( (typestr, namestr, '<ommited>') )
                continue
            valstr  = printableVal(val,type_bit=type_bit, justlength=justlength)
            if len(valstr) > max_valstr:
                valstr = valstr[0:max_valstr/2]+valstr[-max_valstr/2:-1]
            attri_list.append( (typestr, namestr, valstr) )    
        attri_list.sort()
        for (typestr, namestr, valstr) in attri_list:
            entrytail = '\n' if valstr.count('\n') <= 1 else '\n\n'
            typestr2 = typestr+' ' if type_bit else ''
            body += typestr2 + namestr + ' = ' + valstr + entrytail
        return body

    def format_printable(self, type_bit=False, indstr='  * '):
        _printable_str = self.get_printable(type_bit=type_bit)
        _printable_str = _printable_str.replace('\r','\n')
        _printable_str = indstr+_printable_str.strip('\n').replace('\n','\n'+indstr)
        return _printable_str
#---------------
def printableType(val, name=None, parent=None):
    if hasattr(parent, 'customPrintableType'):
        # Hack for non-trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr != None:
            return _typestr
    if type(val) == np.ndarray:
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type','')
        _typestr = re.sub('[\'><]','',_typestr)
        _typestr = re.sub('  *',' ',_typestr)
        _typestr = _typestr.strip()
    return _typestr
#---------------
def printableVal(val,type_bit=True, justlength=False):
    # NUMPY ARRAY
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr == 'bool':
            _valstr = '{ shape:'+info.shapestr+' bittotal: '+info.bittotal+'}'# + '\n  |_____'
        else: 
            _valstr = '{ shape:'+info.shapestr+' mM:'+info.minmaxstr+' }'# + '\n  |_____'
    # String
    elif type(val) is types.StringType:
        _valstr = '\'%s\'' % val
    # List
    elif type(val) is types.ListType:
        if justlength or len(val) > 30:
            _valstr = 'len='+str(len(val))
        else:
            _valstr = '[ '+(',\n  '.join([str(v) for v in val]))+' ]'
    elif hasattr(val, 'get_printable') and type(val) != type: #WTF? isinstance(val, AbstractPrintable):
        _valstr = val.get_printable(type_bit=type_bit)
    elif type(val) is types.DictType:
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  '+str(val_key) + ' : ' + str(val_val)+'\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0: # Indent if necessary
        _valstr = _valstr.replace('\n','\n    ')
        _valstr = '\n    '+_valstr
    _valstr = re.sub('\n *$','', _valstr) # Replace empty lines 
    return _valstr
#---------------

class DynStruct(AbstractPrintable):
    ' dynamical add and remove members '
    def __init__(self, child_exclude_list=[], copy_dict=None, copy_class=None):
        super(DynStruct, self).__init__(child_exclude_list)
        if type(copy_dict) == types.DictType:
            self.add_dict(copy_dict)

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

#---------------
def npArrInfo(arr):
    info = DynStruct()
    info.shapestr  = '['+' x '.join([str(x) for x in arr.shape])+']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1-arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0: 
            info.minmaxstr = '(%r,%r)' % (arr.min(), arr.max())
        else: 
            info.minmaxstr = '(None)'
    return info

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('[???] __main__ = Printable.py')
