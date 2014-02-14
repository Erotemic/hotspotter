from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[dev_api]', DEBUG=False)
import dev_stats
import dev_consistency
import dev_reload
from hscom.Printable import DynStruct


class DebugAPI(DynStruct):
    def __init__(dbg, hs):
        super(DebugAPI, dbg).__init__(child_exclude_list=['hs'])
        dbg.hs = hs

    def dbstats(dbg):
        return dev_stats.dbstats(dbg.hs)

    def detect_duplicate_images(dbg):
        return dev_consistency.detect_duplicate_images(dbg.hs)

    def check_keypoint_consistency(dbg):
        return dev_consistency.check_keypoint_consistency(dbg.hs)

    def reload(dbg):
        dev_reload.reload_all_modules()


def augment_api(hs):
    dbg = DebugAPI(hs)
    hs.dbg = dbg
    return hs
