from hotspotter.other.logger  import logmsg, logerr
from hotspotter.other.helpers import alloc_lists
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.QueryManager import QueryResult
from pylab import find
from numpy import setdiff1d, ones, zeros, array, int32
import shelve

class ExperimentManager(AbstractManager):
    def __init__(em, hs):
        super(ExperimentManager, em).__init__(hs)
        em.cx2_res = None
        em.recompute_bit = False


    def print_matching_results(em): 
        iom = em.hs.iom
        em.run_singleton_queries()
        matching_image_list = []
        for res in enumerate(em.cx2_res):
            pass



    def run_experiment(em):
        iom = em.hs.iom
        em.run_singleton_queries()
        report_str = em.get_report_str()
        report_fpath = iom.get_temp_fpath('expt_report_'+em.get_expt_suffix()+'.txt')
        with open(report_fpath,'w') as f:
            f.write(report_str)
        print report_str

    def display_results(em, fname=None):
        cm, iom = em.hs.get_managers('cm','iom')
        fname= r'expt_match_list.samp1.algo.5.txt'
        if fname is None: fname = 'expt_match_list'+em.get_expt_suffix()+'.txt'
        fpath = iom.get_temp_fpath(fname)
        list_matches_file = iom.get_temp_fpath(fpath)
        num_show = 4
        cid_list = [0]*num_show
        titles = [None]*num_show
        import os
        try:
            os.makedirs(iom.get_temp_fpath('imgres'))
        except Exception:
            pass
        with open(list_matches_file, 'r') as file:
            file = open(list_matches_file, 'r')
            file.seek(0)
            for line in file:
                line = line.replace('\n\n','\nSLASHN')
                line = line.replace('\r','\n').replace('\n\n','\n')
                line = line.replace('\nSLASHN','\n\n')
                if line.strip(' ') == '\n':
                    save_fname = 'imgres/sim=%07.2f-qcid=%d.png' % (maxsim, cid_list[0])
                    kwargs = {
                        'titles' : titles,\
                        'save_fpath' : iom.get_temp_fpath(save_fname),\
                        'fignum' : 1
                    }
                    em.hs.show_chips(cid_list, **kwargs)
                    continue
                fields = line.replace('\n','').split(',')
                if fields[0] == 'QUERY':
                    rank = 0
                    scorestr = 'QUERY'
                else:
                    rank = int(fields[0].replace('rank=',''))
                    score = fields[3].replace('score=','').strip(' ')
                    if rank == 1:
                        maxsim = float(score)
                    scorestr = 'SCORE = '+score
                cid_list[rank] = int(fields[2].replace('cid=',''))
                titles[rank] = scorestr


    def list_matches(em):
        '''Quick experiment:
           Query each chip with a duplicate against whole database
           Do not remove anyone from ANN matching'''
        hs = em.hs
        cm, vm, qm = hs.get_managers('cm','vm', 'qm')
        logmsg('Running List Matches Experiment')
        vm.build_model()
        list_matches_file = iom.get_temp_fpath('expt_match_list'+em.get_expt_suffix()+'.txt')
        with open(list_matches_file, 'a') as file:
            for cx in iter(cm.get_valid_cxs()):
                res = qm.cx2_res(cx)
                cid, gname = cm.cx2_(res.rr.qcx, 'cid', 'gname')
                (tcid , tgname  , tscore ) = res.tcid2_('cid','gname','score')
                logmsg('---QUERY---')
                outstr = 'QUERY,    gname=%s, cid=%4d' % (gname, cid)
                print outstr 
                file.write(outstr+'\n')
                for (rank, tup) in enumerate(zip(*[x.tolist() for x in (tgname, tcid, tscore )])):
                    outstr = '  rank=%d, gname=%s, cid=%4d, score=%7.2f' % tuple([rank+1]+list(tup))
                    print outstr 
                    file.write(outstr+'\n')
                print ''
                file.write('\n\n')

    def run_singleton_queries(em):
        '''Quick experiment:
           Query each chip with a duplicate against whole database
           Do not remove anyone from ANN matching'''
        hs = em.hs
        cm, vm = hs.get_managers('cm','vm')
        valid_cx = cm.get_valid_cxs()
        cx2_num_other = cm.cx2_num_other_chips(valid_cx)
        singleton_cx = valid_cx[cx2_num_other == 1] # find singletons
        duplicate_cx = valid_cx[cx2_num_other  > 1] # find matchables
        cx2_rr = em.batch_query(force_recomp=em.recompute_bit, test_cxs=duplicate_cx)
        em.cx2_res = array([  [] if rr == [] else\
                           QueryResult(hs,rr) for rr in cx2_rr])

    def get_expt_suffix(em):
        vm, am = em.hs.get_managers('vm','am')
        samp_suffix = vm.get_samp_suffix()
        algo_suffix = am.get_algo_suffix(depends='all')
        return samp_suffix+algo_suffix


    def batch_query(em, force_recomp=False, test_cxs=None):
        'TODO: Fix up the VM dependencies'
        vm, iom, am = em.hs.get_managers('vm','iom','am')
        shelf_fpath = iom.get_temp_fpath('qres_shelf'+em.get_expt_suffix()+'.db')
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
        em.hs.dm.show_query(res)

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

    def get_report_str(em):
        '''TODO: I want to see: number of matches, the score of each type of matcher
           I want to see this in graph format, but first print format. I want to see 
           the precision and recall. I want to see the rank of the best, I want to see
           the rank of the worst minus the number of correct answers. Once we have the 
           metrics, we can build the learners, visualizations, and beter algorithms. 

           Average Num descriptors per image / StdDev / Min / and Max
        '''
        report_str = ''

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

        report_str += 'WORST QUERY RESULTS:'+'\n'
        report_str += 'GT C AVE-RANK  | QCID - GT_CIDS, GT_CIDS_RANK'+'\n'
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
            report_str += '%14.3f | %4d - %s' % (score, cid, str(zip(other_cids, array(other_rank, dtype=int32))))+'\n'


        report_str += '\nGT N RANK      | QNID | QCID - GT_CIDS'+'\n'
        for score, cx in worst_nids:
            if score < 0: continue        
            cid = cm.cx2_cid[cx]
            nid = cm.cx2_nid(cx)
            other_cids = setdiff1d(cm.cx2_cid[cm.cx2_other_cxs([cx])[0]], cid)
            report_str +=  '%14.3f | %4d | %4d - %s' % (score, nid, cid, str(other_cids))+'\n'

        report_str += '\n\nOVERALL WORST NAMES:'+'\n'
        worst_names = zip(nid2_badness, range(len(nid2_badness)))
        worst_names.sort()
        report_str += 'SUM C AVE RANK | NID - CIDS'+'\n'
        for score, nid in worst_names:
            if score < 0: continue
            nx = nm.nid2_nx[nid]
            name = nm.nx2_name[nx]
            other_cids = cm.cx2_cid[nm.nx2_cx_list[nx]]
            report_str += ' %13.3f | %4d | %s' % (score, nid, str(other_cids))+'\n'

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

            report_str +=  '\n--------\nQueries with chip-rank <= '+str(num_top)+':'+'\n'
            report_str +=  '  #good = %d, #bad =%d, #total=%d' % (num_good_chip, num_bad_chip, total_chip)+'\n'

            report_str +=  'Queries with name-rank <= '+str(num_top)+':'+'\n'
            report_str +=  '  #good = %d, #bad =%d, #total=%d' % (num_good_name, num_bad_name, total_name)+'\n'
    
        report_str +=  '-----'
        report_str +=  'Database: '+am.hs.db_dpath+'\n'
        report_str +=  am.get_algo_name('all')+'\n'
        return report_str
