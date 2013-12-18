import numpy as np
import types
import sys

VALID_INT_TYPES = (types.IntType,
                   types.LongType,
                   np.typeDict['int64'],
                   np.typeDict['int32'],
                   np.typeDict['uint8'],)

VALID_FLOAT_TYPES = (types.FloatType,
                     np.typeDict['float64'],
                     np.typeDict['float32'],
                     np.typeDict['float16'],)

DEBUG = False

if DEBUG:
    def printDBG(msg):
        print('[tools.DBG] '+msg)
else:
    def printDBG(msg):
        pass


def index_of(item, array):
    'index of [item] in [array]'
    return np.where(array == item)[0][0]


def class_iter_input(func):
    def iter_wrapper(self, input_, *args, **kwargs):
        is_scalar = not np.iterable(input_) or is_str(input_)
        result = func(self, (input_,), *args, **kwargs) if is_scalar else \
            func(self, input_, *args, **kwargs)
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


def assert_int(var, lbl='var'):
    try:
        assert is_int(var), 'type(%s)=%r is not int' % (lbl, get_type(var))
    except AssertionError:
        print('[tools] %s = %r' % (lbl, var))
        print('[tools] VALID_INT_TYPES: %r' % VALID_INT_TYPES)
        raise

if sys.platform == 'win32':
    # Well this is a weird system specific error
    # https://github.com/numpy/numpy/issues/3667
    def get_type(var):
        'Gets types accounting for numpy'
        return var.dtype if isinstance(var, np.ndarray) else type(var)
else:
    def get_type(var):
        'Gets types accounting for numpy'
        return var.dtype.type if isinstance(var, np.ndarray) else type(var)



def is_type(var, valid_types):
    'Checks for types accounting for numpy'
    #printDBG('checking type var=%r' % (var,))
    #var_type = type(var)
    #printDBG('type is type(var)=%r' % (var_type,))
    #printDBG('must be in valid_types=%r' % (valid_types,))
    #ret = var_type in valid_types
    #printDBG('result is %r ' % ret)
    return get_type(var) in valid_types


def is_int(var):
    return is_type(var, VALID_INT_TYPES)


def is_float(var):
    return type(var) in VALID_FLOAT_TYPES


def is_str(var):
    return type(var) in VALID_FLOAT_TYPES


def is_bool(var):
    return isinstance(var, bool)
    #return type(var) in VALID_BOOLEAN_TYPES


def is_dict(var):
    return isinstance(var, dict)
    #return type(var) in VALID_BOOLEAN_TYPES


def is_list(var):
    return isinstance(var, list)
    #return type(num) in VALID_LIST_TYPES
