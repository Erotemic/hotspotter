from __future__ import division, print_function
import __builtin__
import sys


def init(module_name, module_prefix='[???]'):
    module = sys.modules[module_name]

    try:
        profile  # NoQA
    except NameError:
        profile = lambda func: func

    def rrr():
        'Dynamic module reloading'
        import imp
        print(module_prefix + ' reloading ' + module_name)
        imp.reload(module)

    def print_(msg):
        sys.stdout.write(msg)

    def print(msg):
        __builtin__.print(msg)

    def noprint(msg):
        pass

    def print_on():
        module.print = print
        module.print_ = print_

    def print_off():
        module.print = noprint
        module.print_ = noprint

    return print, print_, print_on, print_off, rrr, profile
