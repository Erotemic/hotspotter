from __future__ import division, print_function
import match_chips3 as mc3
import matching_functions as mf
import DataStructures as ds
import investigate_chip as iv
import voting_rules2 as vr2
import draw_func2 as df2
import helpers as helpers
import numpy as np
import sys

# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed


# What happens when we take all other possible ground truth matches out
# of the database index? 

#mc3.rrr(); mf.rrr(); iv.rrr(); ds.rrr(); vr2.rrr()
def silent(*args):
    pass
df2.print = silent
vr2.print = silent
ds.print = silent

def prepare_test(hs, **kwargs):
    kwargs = vars().get('kwargs', {})
    q_params = mc3.prequery(hs, **kwargs)
    id2_qcxs, id2_ocids, id2_notes = iv.get_all_history(hs.args.db, hs)
    return q_params, zip(id2_qcxs, id2_ocids, id2_notes)

def compare_scoring(hs):
    q_params, id2_qon = prepare_test(hs, K=5, lnbnn_weight=0)
    id2_bestranks = []
    algos = []
    id2_title = []
    for id_ in xrange(len(id2_qon)):
        (qcx, ocids, notes) = id2_qon[id_]
        gt_cxs = hs.get_other_cxs(qcx)
        title = 'q'+ hs.cxstr(qcx) + ' - ' + notes
        print(title)
        print('gt_'+hs.cxstr(gt_cxs))
        res_list = mc3.execute_query_safe(hs, q_params, [qcx])
        bestranks = []
        for qcx2_res in res_list:
            res = qcx2_res[qcx]
            print('res.title =: '+res.title)
            print('res.uid =: '+res.uid)
            gt_ranks = res.get_gt_ranks(gt_cxs)
            bestranks += [min(gt_ranks)]
        id2_title += [title]
        id2_bestranks += [bestranks]
    id2_title     = np.array(id2_title)
    id2_bestranks = np.array(id2_bestranks)
    id2_title.shape = (len(id2_title), 1)
    print('\n----------')
    #print(id2_bestranks)
    #print(id2_title)
    print(np.hstack([id2_bestranks, id2_title]))

    #execute_query_safe(

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main_locals = iv.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
    compare_scoring(hs)
    #for vals in vr2.TMP:
        #print('-----')
        #for val in vals:
            #print(val)
        #print('-----')
    exec(df2.present())
