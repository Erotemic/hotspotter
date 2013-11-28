import numpy as np
import re
import types
#---------------
def npArrInfo(arr):
    from ConcretePrintable import DynStruct
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
        info.minmaxstr = '(%s,%s)' % ( str( arr.min() if len(arr) > 0 else None ), str( arr.max() if len(arr) > 0 else None ) )
    return info
#---------------
_lbl2_header = {
        'cid'  : 'ChipID'      ,\
        'nid'  : 'NameID'      ,\
        'gid'  : 'ImgID'       ,\
        'roi'  : 'roi[tl_x  tl_y  w  h]',\
        'cx'   : 'ChipIndex'   ,\
        'nx'   : 'NameIndex'   ,\
        'gx'   : 'ImageIndex'  ,\
        'cxs'  : 'ChipIndexes' ,\
        'cids' : 'ChipIDs',\
        'name' : 'Name',\
        'gname': 'ImageName',\
        'theta': 'theta',\
        'num_c': 'Num Chips',\
        'aif'  : 'AllIndexesFound',\
    }
def _lbls2_headers(lbls):
    'Converts lookup keys to readable names if possible'
    return [_lbl2_header[l] if l in _lbl2_header.keys() else l for l in lbls]

def _lbls2_maxvals(lbls, hs):
    '''
    Finds the maximum value seen so far in the managers
    Uses this to figure out how big to make column spacing
    If the info doesnt exist, then defaults to spacing of 10
    '''
    cm = hs.cm
    nm = hs.nm
    gm = hs.gm
    _lbl2_maxval = {
        'cid'  : int(cm.max_cid),\
        'aif'  : 2,\
        'nid'  : int(nm.max_nid),\
        'gid'  : int(gm.max_gid),\
        'roi'  :      cm.max_roi,\
        'cx'   : int(cm.max_cx) ,\
        'nx'   : int(nm.max_nx) ,\
        'gx'   : int(gm.max_gx) ,\
        'cxs'  : int(cm.max_cx) ,\
        'cids' :         '',\
        'name' : nm.max_name,\
        'gname': gm.max_gname,\
        'num_c': 10,\
        'theta': 10.0
    }
    return [_lbl2_maxval[l] if l in _lbl2_header.keys() else l for l in lbls]

def lbls2_format(lbls, hs):
    headers = _lbls2_headers(lbls)
    maxvals = _lbls2_maxvals(lbls, hs)
    #A list of (space,format) tuples
    _spcfmt = [_table_fmt(m, h) for m, h in zip(maxvals, headers)]
    header_space_list = [ t[0] for t in _spcfmt ]
    data_format_list  = [ t[1] for t in _spcfmt ]
    head_format_list  = ', '.join(['{:>%d}']*len(lbls)) % tuple(header_space_list)
    header = head_format_list.format(*headers)
    data_format = ', '.join(data_format_list)
    return (header, data_format)

def _table_fmt(max_val, lbl=""):
    '''
    Table Formater: gives you the python string to format your data
    Input:  longest value
    Output: (nSpaces, formatStr)
    '''
    if max_val == 0:
        max_val = 1
    if type(max_val) is types.IntType or type(max_val) == np.uint32:
        spaces = max(int(np.log10(max_val)), len(lbl))+1
        fmtstr = '{:>%dd}' % spaces
    elif type(max_val) is types.FloatType:
        _nDEC = 3
        if _nDEC == 0:
            spaces = max(int(np.log10(max_val)), len(lbl))+1
            fmtstr = '{:>%d.0f}' % (spaces)
        else:
            spaces = max(int(np.log10(max_val))+1+_nDEC, len(lbl))+1
            fmtstr = '{:>%d.%df}' % (spaces, _nDEC)
    elif type(max_val) is types.ListType:
        _SEP    = '  '
        _rBrace = ' ]'
        _lBrace = '[ '
        # Recursively format elements in the list
        _items  = [_table_fmt(x) for x in max_val]
        _spc    = [ t[0] for t in _items]
        _fmt    = [ t[1] for t in _items]
        spaces  = sum(_spc)+((len(_items)-1)*len(_SEP))+len(_rBrace)+len(_lBrace)
        if spaces < len(lbl):
            _lBrace = ' '*(len(lbl)-spaces) + _lBrace
            #raise Exception('The label is expected to be shorter than the list')
        fmtstr  = _lBrace+_SEP.join(_fmt)+_rBrace
    elif type(max_val) is types.StringType:
        spaces = len(max_val)+1
        fmtstr = '{:>%d}' % (spaces) 
    else:
        raise Exception('Unknown Type for '+str(type(max_val))+'\n label:\"'+str(lbl)+'\" max_val:'+str(max_val) )
    return (spaces, fmtstr)
#-----------

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
            print info.dtypestr
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
        # Get the formating string, so the data looks nice
        format_tup = lbls2_format(lbls, self.hs)
        # header formating string
        header     = format_tup[0]
        # data formating string
        dat_format = format_tup[1]
        # Write info on how many we are writing
        ret  = '# NumData %d\n' % valid_xs.size
        ret += '#'+header+'\n'
        # Ensure iterability
        if not np.iterable(valid_xs):
            valid_xs = [valid_xs]

        # Do work. Get the format data for each valid index and 
        # format it into a nice printable string
        for x in iter(valid_xs):
            tup = tuple()
            for lbl in lbls:
                try:
                    # Use x2_lbl property to get what you need
                    val = self.x2_lbl[lbl](x)
                    if type(val) in [np.uint32, np.bool_, np.bool]:
                        val = int(val)
                    if type(val) == types.StringType or val == []:
                        raise TypeError
                    tup = tup+tuple(val)
                except TypeError:
                    tup = tup+tuple([val])
            ret += ' '+dat_format.format(*tup)+'\n'
        return ret
