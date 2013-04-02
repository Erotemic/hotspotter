import numpy as np
import re
import types
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
        print(self)
    def printme2(self, type_bit=True, print_exclude_aug = []):
        print(self.get_printable(type_bit, print_exclude_aug))
    def get_printable(self, type_bit=True, print_exclude_aug = []):
        body = ''
        attri_list = []
        exclude_key_list = self._printable_exclude+print_exclude_aug
        for (key, val) in self.__dict__.iteritems():
            if key in exclude_key_list: continue
            namestr = str(key)
            valstr  = printableVal(val,type_bit=type_bit)
            typestr = printableType(val)
            max_valstr = 10000
            if len(valstr) > max_valstr:
                valstr = valstr[0:max_valstr/2]+valstr[-max_valstr/2:-1]
            attri_list.append( (typestr, namestr, valstr) )
        
        attri_list.sort()
        for (typestr, namestr, valstr) in attri_list:
            entrytail = '\n' if valstr.count('\n') <= 1 else '\n\n'
            typestr2 = typestr+' ' if type_bit else ''
            body += typestr2 + namestr + ' = ' + valstr + entrytail
        return body
#---------------
def printableType(val):
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
def printableVal(val,type_bit=True):
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr == 'bool':
            _valstr = '{ shape:'+info.shapestr+' bittotal: '+info.bittotal+'}'# + '\n  |_____'
        else: 
            _valstr = '{ shape:'+info.shapestr+' mM:'+info.minmaxstr+' }'# + '\n  |_____'


    elif type(val) is types.StringType:
        _valstr = '\'%s\'' % val
    elif type(val) is types.ListType:
        lenstr = str(len(val))
        _valstr = 'Length:'+lenstr
    elif hasattr(val, 'get_printable'): #WTF? isinstance(val, AbstractPrintable):
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
