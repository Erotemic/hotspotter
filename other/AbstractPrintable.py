import numpy as np
import re
import types
from other.helpers import *
#---------------
def npArrInfo(arr):
    try:
        info = DynStruct()
        info.shapestr  = '['+' x '.join([str(x) for x in arr.shape])+']'
        info.dtypestr  = str(arr.dtype)
        if info.dtypestr == 'bool':
            info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1-arr))
        if info.dtypestr == 'object':
            info.minmaxstr = 'NA'
        else:
            info.minmaxstr = '(%s,%s)' % ( str( arr.min() if len(arr) > 0 else None ), str( arr.max() if len(arr) > 0 else None ) )
    except Exception as ex: 
        logmsg(str(ex))
        logerr(str(ex))
    return info
#---------------
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
class AbstractManager(AbstractPrintable):
    def __init__(self, hs, child_print_exclude=[]):
        super(AbstractManager, self).__init__(['hs'] + child_print_exclude)
        self.hs = hs # ref to core HotSpotter
#---------------
class AbstractDataManager(AbstractManager):
    ' Superclass for chip/name/table managers '
    def __init__(self, hs, child_print_exclude=[]):
        super(AbstractDataManager, self).__init__(hs, child_print_exclude+['x2_lbl'])

    def x2_info(self, valid_xs, lbls):
        ''' Used to print out formated information'''
        format_tup = lbls2_format(lbls, self.hs)
        header     = format_tup[0]
        dat_format = format_tup[1]

        ret  = '# NumData %d\n' % valid_xs.size
        ret += '#'+header+'\n'
        if not numpy.iterable(valid_xs):
            valid_xs = [valid_xs]

        for x in iter(valid_xs):
            tup = tuple()
            for lbl in lbls:
                try:
                    val = self.x2_lbl[lbl](x)
                    if type(val) in [numpy.uint32, numpy.bool_, numpy.bool]:
                        val = int(val)
                    if type(val) == types.StringType or val == []:
                        raise TypeError
                    tup = tup+tuple(val)
                except TypeError:
                    tup = tup+tuple([val])
            logdbg('dat_format: '+str(dat_format))
            logdbg('tuple_types: '+str([type(t) for t in tup]))
            logdbg('format tuple: '+str(tup))
            ret += ' '+dat_format.format(*tup)+'\n'
        return ret
