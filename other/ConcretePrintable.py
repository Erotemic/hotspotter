from other.AbstractPrintable import AbstractPrintable
import types

class DynStruct(AbstractPrintable):
    ' dynamical add and remove members '
    def __init__(self, child_exclude_list=[]):
        super(DynStruct, self).__init__(child_exclude_list)
    def to_dict(self):
        ret = {}
        exclude_key_list = self._printable_exclude
        for (key, val) in self.__dict__.iteritems():
            if key in exclude_key_list: continue
            ret[key] = val
        return ret

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
            ret = []
            for k in key:
                ret.append(getattr(self, k))
        else:
            ret = getattr(self, key)
        return ret
#---------------

class PrefStruct(DynStruct):
    'Structure for Creating Preferences'
    def __init__(self, save_fpath=None, copy_dict=None):
        'Creates a pref struct that will save itself to save_fpath if available and have initail members of some dictionary'
        super(PrefStruct, self).__init__(['save_fpath'])
        self.save_fpath = save_fpath
        self.add_dict(copy_dict)

    def add_dict(self, some_dict):
        if type(some_dict) != types.DictType:
            raise Exception('PrefStruct.add_dict expects a dictionary.'+\
                            'Recieved: '+str(type(some_dict)))
        for (key,val) in some_dict.iter_items():
            self.key = val

    def iter_items(self):
        pass

    def write_to_disk():
        pass

    def read_from_disk():
        pass
