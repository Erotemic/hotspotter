from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[dev_api]', DEBUG=False)
import dev_stats
import dev_consistency
import dev_reload
from hscom.Printable import DynStruct
from hscom import util


class DebugAPI(DynStruct):
    def __init__(dbg, hs):
        super(DebugAPI, dbg).__init__(child_exclude_list=['hs'])
        dbg.hs = hs
        for key, func in util.module_functions(dev_consistency):
            printDBG('[devapi] augmenting: ' + str(func))
            dbg.__dict__[key] = lambda *args: func(dbg.hs, *args)

    def dbstats(dbg):
        return dev_stats.dbstats(dbg.hs)

    def reload(dbg):
        _reload()


def augment_api(hs):
    dbg = DebugAPI(hs)

    hs.dbg = dbg
    return hs


def _reload():
    dev_reload.reload_all_modules()
