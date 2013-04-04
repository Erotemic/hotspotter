from other.helpers import *
from other.logger import *
from core.QueryManager import QueryResult
from pylab import find
from numpy import setdiff1d, ones, zeros, array, int32
import shelve

class ExperimentManager(AbstractManager):
    def __init__(em, hs):
        super(ExperimentManager, em).__init__(hs)
        em.cx2_res = None
        em.recompute_bit = False

    def run_singleton_queries(em):
        '''Quick experiment:
           Query each chip with a duplicate against whole database
           Do not remove anyone from ANN matching'''
        hs = em.hs
        cm, vm = hs.get_managers('cm','vm')
        valid_cx = cm.get_valid_cxs()
        cx2_numother = cm.cx2_num_other_chips(valid_cx)
        singleton_cx = valid_cx[cx2_numother == 1] # find singletons
        duplicate_cx = valid_cx[cx2_numother  > 1] # find matchables
        cx2_rr = em.batch_query(force_recomp=em.recompute_bit, test_cxs=duplicate_cx)
        em.cx2_res = array([  [] if rr == [] else\
                           QueryResult(hs,rr) for rr in cx2_rr])

    def run_nightly(em):
        hs = iom.hs
        iom = hs.iom
        datasets = iom.get_dataset_fpath('Naut')
        #vary params
        hs.am.query.method = 'LNRAT'

    def batch_query(em, force_recomp=False, test_cxs=None):
        'TODO: Fix up the VM dependencies'
        vm, iom, am = em.hs.get_managers('vm','iom','am')
        samp_suffix = vm.get_samp_suffix()
        algo_suffix = am.get_algo_suffix(depends='all')
        qres_shelf_fname = 'qres_shelf'+samp_suffix+algo_suffix+'.db'
        shelf_fpath = iom.get_temp_fpath(qres_shelf_fname)
        # Compute the matches
        qm = vm.hs.qm
        vm.build_model(force_recomp=force_recomp)
        if test_cxs == None:
            test_cxs = vm.get_train_cx()
        cx2_rr = alloc_lists(vm.hs.cm.max_cx+1)
        logmsg('Building matching graph. This may take awhile')
        total = len(test_cxs)
        count = 0
        need_to_save = force_recomp
        shelf = shelve.open(shelf_fpath)
        for cx in test_cxs:   
            count+=1
            shelf_key = str(cx)
            rr = None
            if not force_recomp and shelf_key in shelf.keys():
                logmsg('Reloading %d/%d' % (count, total)) 
                try:
                    rr = shelf[str(cx)]
                except Exception: 
                    logmsg('Error reading '+str(cx))
            if rr == None:
                logmsg('Query %d/%d' % (count, total))
                rr = qm.cx2_res(cx).rr
                need_to_save = True
            cx2_rr[cx] = rr
        shelf.close()
        if need_to_save: # Save the matches
            shelf = shelve.open(shelf_fpath)
            try:
                # try and save some memory
                for i in range(len(cx2_rr)):
                    logmsg('SavingRR: '+str(i))
                    to_save = cx2_rr[i]
                    if to_save == []:
                        continue
                    to_save.cx2_cscore_ = []
                    to_save.cx2_fs_ = []
                    to_save.qfdsc = []
                    to_save.qfpts = []
                    shelf[str(to_save.qcx)] = to_save
                shelf.sync()
            except Exception as ex:
                logerr('Error saving to the shelf: '+str(ex))
            finally:
                shelf.close()
        logmsg('Done building matching graph.')
        return cx2_rr

    def show_query(em, cx):
        res = em.cx2_res[cx]
        hs.show_query(res)

    def get_result_rank_histogram(em):
        '''returns a histogram of the number of queries with 
        correct matches with some rank. The rank is the index
        into the histogram'''
        if em.cx2_res is None: 
            logerr('You cant get results on unrun experiments')

        cm,nm,am = em.hs.get_managers('cm','nm','am')
        # gt hist shows how often a chip is at rank X
        # opt is the optimistic rank. Precision
        # pas is the pesemistic rank. Recallish
        rank_hist_opt = zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        rank_hist_pes = zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        for res in em.cx2_res:
            if res == []: continue
            cx = res.rr.qcx
            qnid = res.rr.qnid
            # Evaluate considering the top returned chips and names
            top_cx      = res.cx_sort()
            gt_pos_chip = (1+find(qnid == cm.cx2_nid(top_cx)))
            #Overflow, the last position is past the num_top
            if len(gt_pos_chip) == 0:
                rank_hist_opt[-1] += 1 
                rank_hist_pes[-1] += 1
            else: 
                rank_hist_opt[min(gt_pos_chip)] += 1
                rank_hist_pes[max(gt_pos_chip)-len(gt_pos_chip)+1] += 1
        return rank_hist_opt

    def show_problems(em):
        cm,nm,am = em.hs.get_managers('cm','nm','am')
        # Evaluate rank by chip and rank by name
        cx2_error_chip = -ones(len(em.cx2_res))
        cx2_error_name = -ones(len(em.cx2_res))
        # gt hist shows how often a chip is at rank X
        gt_hist_chip = zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        # FIXME: what if a lot of names are UNIDEN?
        gt_hist_name = zeros(len(nm.get_valid_nxs())+2) # and 1 at the end for overflow
        for res in em.cx2_res:
            if res == []: continue
            cx = res.rr.qcx
            qnid = res.rr.qnid
            # Evaluate considering the top returned chips and names
            top_cx      = res.cx_sort()
            gt_pos_chip = (1+find(qnid == cm.cx2_nid(top_cx)))
            #Overflow, the last position is past the num_top
            if len(gt_pos_chip) == 0: gt_hist_chip[-1] += 1 
            else: gt_hist_chip[min(gt_pos_chip)] += 1
            if len(gt_pos_chip) > 0:
                # Reward more chips for being in the top X
                cx2_error_chip[cx] = float(sum(gt_pos_chip))/len(gt_pos_chip)
            
            (top_nx, _) = res.nxcx_sort()
            gt_pos_name = (1+find(qnid == nm.nx2_nid[top_nx]))
            if len(gt_pos_name) == 0: gt_hist_name[-1] += 1 
            else: gt_hist_name[min(gt_pos_name)] += 1
            if len(gt_pos_name) > 0:
                cx2_error_name[cx] = float(sum(gt_pos_name))/len(gt_pos_name) 

        worst_cids = zip(cx2_error_name, range(len(cx2_error_name)))
        worst_nids = zip(cx2_error_chip, range(len(cx2_error_chip)))
        worst_cids.sort()
        worst_nids.sort()

        nid2_badness = -ones(len(nm.nid2_nx))

        print 'WORST QUERY RESULTS:'
        print 'GT C AVE-RANK  | QCID - GT_CIDS, GT_CIDS_RANK'
        for score, cx in worst_cids:
            if score < 0: continue        
            cid = cm.cx2_cid[cx]
            nid = cm.cx2_nid(cx)
            if nid2_badness[nid] == -1:
                nid2_badness[nid] = 0
            nid2_badness[nid] += score
            other_cxs  = setdiff1d(cm.cx2_other_cxs([cx])[0], [cx])
            top_cx = em.cx2_res[cx].cx_sort()
            other_rank = []
            for ocx in other_cxs:
                to_append = find(top_cx == ocx)+1
                other_rank.extend(to_append.tolist())
            other_cids = cm.cx2_cid[other_cxs]
            #print '%14.3f | %4d - %s,%s %s' % (score, cid, str(other_cids), '\n'+' '*23, str(array(other_rank, dtype=int32)))
            print '%14.3f | %4d - %s' % (score, cid, str(zip(other_cids, array(other_rank, dtype=int32))))


        print '\nGT N RANK      | QNID | QCID - GT_CIDS'
        for score, cx in worst_nids:
            if score < 0: continue        
            cid = cm.cx2_cid[cx]
            nid = cm.cx2_nid(cx)
            other_cids = setdiff1d(cm.cx2_cid[cm.cx2_other_cxs([cx])[0]], cid)
            print '%14.3f | %4d | %4d - %s' % (score, nid, cid, str(other_cids))

        print '\n\nOVERALL WORST NAMES:'
        worst_names = zip(nid2_badness, range(len(nid2_badness)))
        worst_names.sort()
        print 'SUM C AVE RANK | NID - CIDS'
        for score, nid in worst_names:
            if score < 0: continue
            nx = nm.nid2_nx[nid]
            name = nm.nx2_name[nx]
            other_cids = cm.cx2_cid[nm.nx2_cx_list[nx]]
            print ' %13.3f | %4d | %s' % (score, nid, str(other_cids))

        #num less than 5 (chips/names)
        num_tops = [1,5]
        for num_top in num_tops:
            top_chip      = min(num_top+1,len(gt_hist_chip)-1)
            num_chips     = len(gt_hist_chip)-2
            num_good_chip = sum(gt_hist_chip[0:top_chip])
            num_bad_chip  = sum(gt_hist_chip[top_chip:])
            total_chip    = sum(gt_hist_chip)

            top_name      = min(num_top+1,len(gt_hist_name)-1)
            num_names     = len(gt_hist_name)-2
            num_good_name = sum(gt_hist_name[0:top_name])
            num_bad_name  = sum(gt_hist_name[top_name:])
            total_name    = sum(gt_hist_name)

            print '\n--------\nQueries with chip-rank <= '+str(num_top)+':'
            print '  #good = %d, #bad =%d, #total=%d' % (num_good_chip, num_bad_chip, total_chip)

            print 'Queries with name-rank <= '+str(num_top)+':'
            print '  #good = %d, #bad =%d, #total=%d' % (num_good_name, num_bad_name, total_name)
    
        print '-----'
        print am.get_algo_name('all')
