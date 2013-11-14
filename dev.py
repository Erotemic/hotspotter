import match_chips3 as mc3
import matching_functions as mf
import DataStructures as ds
import investigate_chip as iv
import voting_rules2 as vr2
import draw_func2 as df2
from match_chips3 import *
from matching_functions import *

# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed



mc3.rrr(); mf.rrr(); iv.rrr(); ds.rrr(); vr2.rrr()

def prepare_test(hs, **kwargs):
    kwargs = vars().get('kwargs', {})
    query_params = mc3.prequery(hs, **kwargs)
    id2_qcxs, id2_ocids, id2_notes = iv.get_all_history(hs.args.db, hs)
    return query_params, zip(id2_qcxs, id2_ocids, id2_notes)

def compare_scoring(hs):
    query_params, id2_qon = prepare_test(hs, K=1, lnbnn_weight=1)
    id2_bestranks = []
    for id_ in xrange(len(id2_qon)):
        (qcx, ocids, notes) = id2_qon[id_]
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        print(title)
        reses = mc3.execute_query_safe(hs, query_params, [qcx])
        gt_cxs = hs.get_other_cxs(qcx)
        res = reses[2][qcx]
        cx2_score = res.get_cx2_score(hs)
        top_cxs  = cx2_score.argsort()[::-1]
        gt_ranks = [helpers.npfind(top_cxs == gtcx) for gtcx in gt_cxs]
        bestrank = min(gt_ranks)
        id2_bestranks += [bestrank]
    print(id2_bestranks)
    #execute_query_safe(

main_locals = iv.main()
execstr = helpers.execstr_dict(main_locals, 'main_locals')
exec(execstr)
compare_scoring(hs)
exec(df2.present())
