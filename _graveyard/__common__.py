from __future__ import division, print_function
import __builtin__
import sys

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on(): global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off(): global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[___] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

code = '''
from __future__ import division, print_function
import __builtin__
import sys

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[___] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module
'''
