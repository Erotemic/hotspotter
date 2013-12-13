import numpy as np
import types


VALID_INT_TYPES = set((types.IntType,
                   types.LongType,
                   np.typeDict['int64'],
                   np.typeDict['int32'],
                   np.typeDict['uint8'],
                   ))
VALID_FLOAT_TYPES = set((types.FloatType,
                     np.typeDict['float64'],
                     np.typeDict['float32'],
                     np.typeDict['float16']))

VALID_STRING_TYPES = set((types.StringType,))

def index_of(item, array):
    'index of [item] in [array]'
    return np.where(array==item)[0][0]

def class_iter_input(func): 
    def iter_wrapper(self, input_, *args, **kwargs):
        is_scalar = not np.iterable(input_) or is_str(input_)
        result = func(self, (input_,), *args, **kwargs) if is_scalar else\
                 func(self,  input_,   *args, **kwargs)
        if is_scalar:
            result = result[0]
        return result
    iter_wrapper.func_name = func.func_name
    return iter_wrapper

def debug_exception(func): 
    def ex_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as ex:
            print('[tools] ERROR: %s(%r, %r)' % (func.func_name, args, kwargs))
            print('[tools] ERROR: %r' % ex)
            raise
    ex_wrapper.func_name = func.func_name
    return ex_wrapper

def is_int(num):
    return type(num) in VALID_INT_TYPES
def is_float(num):
    return type(num) in VALID_FLOAT_TYPES
def is_str(num):
    return type(num) in VALID_FLOAT_TYPES

