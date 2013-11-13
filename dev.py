import match_chips3 as mc3
import matching_functions as mf
import DataStructures as ds
import investigate_chip as iv
import voting_rules2 as vr2
from match_chips3 import *
from matching_functions import *

mc3.rrr(); mf.rrr(); iv.rrr(); ds.rrr(); vr2.rrr()

def prepare_test(hs, **kwargs):
    kwargs = vars().get('kwargs', {})
    query_params = mc3.prequery(hs, **kwargs)
    id2_qcxs, id2_ocids, id2_notes = iv.get_all_history(hs.args.db, hs)
    return query_params, zip(id2_qcxs, id2_ocids, id2_notes)

def compare_scoring(hs):
    query_params, id2_qon = prepare_test(hs, K=20, lnbnn_weight=1)
    for id_ in xrange(len(id2_qon)):
        (qcx, ocids, notes) = id2_qon[id_]
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        print(title)
        #execute_query_safe(
