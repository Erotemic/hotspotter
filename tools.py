import numpy as np
import types

VALID_INT_TYPES = set((types.IntType,
                       types.LongType,
                       np.typeDict['int64'],
                       np.typeDict['int32'],
                       np.typeDict['uint8'],))

VALID_FLOAT_TYPES = set((types.FloatType,
                         np.typeDict['float64'],
                         np.typeDict['float32'],
                         np.typeDict['float16'],))

VALID_STRING_TYPES = set((types.StringType,))

#VALID_LIST_TYPES = set((types.ListType,))

#VALID_BOOLEAN_TYPES = set((types.BooleanType,))


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
        assert is_int(var), 'type(%s)=%r =? INT' % (lbl, gettype(var))
    except AssertionError:
        print('[tools] VALID_INT_TYPES: %r' % VALID_INT_TYPES)
        raise


def gettype(var):
    'Gets types accounting for numpy'
    return var.dtype.type if isinstance(var, np.ndarray) else type(var)


def istype(var, valid_types):
    'Checks for types accounting for numpy'
    return gettype(var) in valid_types


def is_int(var):
    printDBG('Checking type: type(var) = %r ' % gettype(var))
    return istype(var, VALID_INT_TYPES)


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
