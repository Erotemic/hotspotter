from other.helpers import *
import types

class PrefStruct(DynStruct):
    'Structure for Creating Preferences'
    def __init__(self, save_fpath=None, copy_dict=None):
        'Creates a pref struct that will save itself to save_fpath if available and have initail members of some dictionary'
        super(PrefStruct, self).__init__(['save_fpath'])
        self.save_fpath = save_fpath
        self.add_dict(copy_dict)

    def add_dict(some_dict):
        if type(some_dict) != types.DictType:
            raise Exception('PrefStruct.add_dict expects a dictionary.'+\
                            'Recieved: '+str(type(some_dict)))
        for (key,val) in copy_dict.iter_items():
            self.key = val

    def iter_items():
        pass

    def write_to_disk():
        pass

    def read_from_disk():
        pass
