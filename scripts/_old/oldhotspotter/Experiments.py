from hotspotter.helpers import alloc_lists
from hotspotter.other.logger  import logmsg, logerr, logwarn
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.QueryManager import QueryResult
import cPickle
import os
import numpy as np
import pylab
import shelve

class ExperimentManager(AbstractManager):
    def __init__(em, hs):
        super(ExperimentManager, em).__init__(hs)
        em.cx2_res = None
        em.recompute_bit = False


    def run_name_consistency_experiment(em):
        iom = em.hs.iom
        # Run a batch query
        em.run_singleton_queries()
        # Get name consistency report of the run query
        report_str = em.get_name_consistency_report_str()
        timestamp  = iom.get_timestamp()
        report_fname = 'expt_name_consistency_report_'+\
                em.get_expt_suffix()+'_'+timestamp+'.txt'
        report_fpath = iom.get_user_fpath(report_fname)
        with open(report_fpath,'w') as f:
            f.write(report_str)
        print report_str


    def run_matching_experiment(em, expt_name='MatchingExperiment',
                                with_images=True):
        '''Quick experiment:
           Query each chip with a duplicate against whole database
           Do not remove anyone from ANN matching'''
        import os
        logmsg('Running Quick Experiment: '+expt_name)
        hs = em.hs
        am, cm, vm, qm, iom, dm = hs.get_managers('am', 'cm','vm', 'qm', 'iom', 'dm')
        with_ellipses = True
        with_points   = False
        with_full_results = False
        with_ellipse_and_points  = False
        # Create an Experiment Directory in .hs_internals/computed
        timestamp  = iom.get_timestamp()
        expt_dpath = iom.ensure_directory(iom.get_user_fpath('_'.join(['expt',expt_name,timestamp])))
        expt_img_dpath = iom.ensure_directory(os.path.join(expt_dpath, 'result_images'))
        expt_rr_dpath  = iom.ensure_directory(os.path.join(expt_dpath, 'raw_results'))

        # Write Algorithm Settings to the file
        algo_prefs_text = am.get_algo_name('all')
        algo_prefs_fpath = os.path.join(expt_dpath, 'algo_prefs.txt')
        iom.write(algo_prefs_fpath, algo_prefs_text)
        
        vm.build_model() # Defaults to building model of all

        prev_ell = dm.draw_prefs.ellipse_bit
        prev_pts = dm.draw_prefs.points_bit

        dm.fignum = 1

        # Create Matches File To Append to
        with open(os.path.join(expt_dpath,'matches.txt'), 'a') as file:
            for cx in iter(cm.get_valid_cxs()):
                # Preform Query
                res = QueryResult(qm.hs, qm.cx2_rr(cx))
                # Get Query Info
                qcid, gname = cm.cx2_(res.rr.qcx, 'cid', 'gname')
                # Get Result Info
                (tcid , tgname  , tscore ) = res.tcid2_('cid','gname','score')
                # Print Query Info
                logmsg('---QUERY---')
                outstr = 'QUERY,    gname=%s, cid=%4d' % (gname, qcid)
                print outstr 
                file.write(outstr+'\n')
                # Print Result Info
                if len(tscore) == 0:
                    maxsim = 0 # Best Score
                if len(tscore) > 0:
                    maxsim = tscore[0] # Best Score
                for (rank, tup) in enumerate(zip(*[x.tolist() for x in (tgname, tcid, tscore )])):
                    outstr = '  rank=%d, gname=%s, cid=%4d, score=%7.2f' % tuple([rank+1]+list(tup))
                    print outstr 
                    file.write(outstr+'\n')
                print ''
                file.write('\n\n')
                # Output Images
                query_name = 'sim=%07.2f-qcid=%d.png' % (maxsim, qcid)
                if with_full_results:
                    #import shelve
                    #shelve.open(os.path.join(expt_rr_dpath, 'rr_'+query_name+'.shelf')
                    #import cPickle
                    #rr = res.rr
                    #cPickle.dump(, rr)
                    pass
                if with_images:
                    dm.draw_prefs.ellipse_bit = False
                    dm.draw_prefs.points_bit  = False
                    dm.draw_prefs.bbox_bit = False
                    dm.show_query(res)
                    dm.save_fig(os.path.join(expt_img_dpath, 'img_'+query_name))
                    if with_ellipses:
                        dm.draw_prefs.ellipse_bit = True
                        dm.draw_prefs.points_bit  = False
                        dm.show_query(res)
                        dm.save_fig(os.path.join(expt_img_dpath, 'ellipse_'+query_name))
                    if with_points:
                        dm.draw_prefs.ellipse_bit = False
                        dm.draw_prefs.points_bit  = True
                        dm.show_query(res)
                        dm.save_fig(os.path.join(expt_img_dpath, 'point_'+query_name))
                    if with_ellipse_and_points:
                        dm.draw_prefs.ellipse_bit = True
                        dm.draw_prefs.points_bit  = True
                        dm.show_query(res)
                        dm.save_fig(os.path.join(expt_img_dpath, 'both_'+query_name))
        logmsg('Finished Matching Experiment: '+expt_name)
        timestamp  = iom.get_timestamp()
        iom.write(os.path.join(expt_dpath, 'Finished_'+str(timestamp)), timestamp+'\n'+em.hs.get_database_stat_str())
        prev_ell = dm.draw_prefs.ellipse_bit
        prev_pts = dm.draw_prefs.points_bit


    def run_singleton_queries(em):
        '''Quick experiment:
           Query each chip with a duplicate against whole database
           Do not remove anyone from ANN matching'''
        cm, vm = em.hs.get_managers('cm','vm')
        valid_cx = cm.get_valid_cxs()
        cx2_num_other = cm.cx2_num_other_chips(valid_cx)
        singleton_cx = valid_cx[cx2_num_other == 1] # find singletons
        duplicate_cx = valid_cx[cx2_num_other  > 1] # find matchables
        cx2_rr = em.batch_query(force_recomp=em.recompute_bit, test_cxs=duplicate_cx)
        em.cx2_res = np.array([  [] if rr == [] else\
                           QueryResult(em.hs,rr) for rr in cx2_rr])

    def run_all_queries(em):
        cm, vm = em.hs.get_managers('cm','vm')
        all_cxs = cm.get_valid_cxs()
        cx2_rr = em.batch_query(force_recomp=em.recompute_bit, test_cxs=all_cxs)
        em.cx2_res = np.array([  [] if rr == [] else\
                           QueryResult(em.hs,rr) for rr in cx2_rr])
        pass




    def threaded_run_and_save_queries(query_cxs, rr_fmtstr_cid):
        # UNTESTED
        import thread
        from threading import Thread
        class run_threaded_queries(Thread):
            def __init__ (self, cx, rr_fmtstr_cid):
                Thread.__init__(self)
                self.cx = cx
                self.rr_fmtstr_cid = rr_fmtstr_cid
            def run(self):
                mutex.acquire()
                output.append(em.run_and_save_query(self.cx, self.rr_fmtstr_cid))
                mutex.release()  
        threads = []
        output = []
        mutex = thread.allocate_lock()
        for cx in iter(query_cxs):
            current = run_threaded_queries(cx, rr_fmtstr_cid)
            threads.append(current)
            current.start()
        for t in threads:
            t.join()
        


    def run_and_save_query(em, cx, rr_fmtstr_cid):
        cid = em.hs.cm.cx2_cid[cx]
        rr_fpath = rr_fmtstr_cid % cid
        rr = em.hs.qm.cx2_rr(cx)
        rr_file = open(rr_fpath, 'wb')
        cPickle.dump(rr, rr_file)
        rr_file.close()
        pass



    def batch_query(em, force_recomp=False, test_cxs=None):
        '''Runs each test_cxs as a query. If test_cxs is None, then all queries
        are run'''
        'TODO: Fix up the VM dependencies'
        vm, iom, am, cm = em.hs.get_managers('vm','iom','am', 'cm')
        # Compute the matches
        qm = vm.hs.qm
        vm.sample_train_set()
        vm.build_model(force_recomp=force_recomp)
        if test_cxs == None:
            test_cxs = vm.get_train_cx()
        logmsg('Building matching graph. This may take awhile')

        depends = ['chiprep','preproc','model','query']
        algo_suffix = am.get_algo_suffix(depends)
        samp_suffix = vm.get_samp_suffix()
        result_dpath = iom.ensure_directory(iom.get_temp_fpath('raw_results'))
        rr_fmtstr_cid = os.path.join(result_dpath, 'rr_cid%07d'+samp_suffix+algo_suffix+'.pkl')

        # Find the Queries which need to be run
        unsaved_cxs = []
        for cx in iter(test_cxs):
            cid = cm.cx2_cid[cx]
            rr_fpath = rr_fmtstr_cid % cid
            if not os.path.exists(rr_fpath):
                unsaved_cxs.append(cx)
        
        # Run Unsaved Query
        total = len(unsaved_cxs)
        for count, cx in enumerate(unsaved_cxs):   
            logmsg('Query %d/%d' % (count, total))
            em.run_and_save_query(cx, rr_fmtstr_cid)

        # Read Each Query 
        cx2_rr = alloc_lists(test_cxs.max()+1)
        total = len(test_cxs)
        for count, cx in enumerate(test_cxs):
            logmsg('Loading Result %d/%d' % (count, total))
            cid = cm.cx2_cid[cx]
            rr_fpath = rr_fmtstr_cid % cid
            if not os.path.exists(rr_fpath):
                logwarn('Result does not exist for CID=%d' % cid)
            rr_file = open(rr_fpath,'rb')
            try: 
                rr = cPickle.load(rr_file)
            except EOFError:
                rr_file.close()
                os.remove(rr_fpath)
                logwarn('Result was corrupted for CID=%d' % cid)

            rr_file.close()
            rr.cx2_cscore_ = []
            rr.cx2_fs_ = []
            rr.qfdsc = []
            rr.qfpts = []
            cx2_rr[cx] = rr

        return cx2_rr


    def get_expt_suffix(em):
        vm, am = em.hs.get_managers('vm','am')
        samp_suffix = vm.get_samp_suffix()
        algo_suffix = am.get_algo_suffix(depends='all')
        return samp_suffix+algo_suffix

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
        rank_hist_opt = np.zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        rank_hist_pes = np.zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        for res in em.cx2_res:
            if res == []: continue
            cx = res.rr.qcx
            qnid = res.rr.qnid
            # Evaluate considering the top returned chips and names
            top_cx      = res.cx_sort()
            gt_pos_chip = (1+pylab.find(qnid == cm.cx2_nid(top_cx)))
            #Overflow, the last position is past the num_top
            if len(gt_pos_chip) == 0:
                rank_hist_opt[-1] += 1 
                rank_hist_pes[-1] += 1
            else: 
                rank_hist_opt[min(gt_pos_chip)] += 1
                rank_hist_pes[max(gt_pos_chip)-len(gt_pos_chip)+1] += 1
        return rank_hist_opt

    # Score a single query for name consistency
    # Written: 5-28-2013 
    def res2_name_consistency(em, res):
        '''Score a single query for name consistency
        Input: 
            res - query result
        Returns: Dict
            error_chip - degree of chip error
            name_error - degree of name error
            gt_pos_name - 
            gt_pos_chip - 
        '''
        # Defaults to -1 if no ground truth is in the top results
        cm, nm = em.hs.get_managers('cm','nm')
        qcx  = res.rr.qcx
        qnid = res.rr.qnid
        qnx   = nm.nid2_nx[qnid]
        ret = {'name_error':-1, 'chip_error':-1,
               'gt_pos_chip':-1, 'gt_pos_name':-1, 
               'chip_precision': -1, 'chip_recall':-1}
        if qnid == nm.UNIDEN_NID: exec('return ret')
        # ----
        # Score Top Chips
        top_cx = res.cx_sort()
        gt_pos_chip_list = (1+pylab.find(qnid == cm.cx2_nid(top_cx)))
        # If a correct chip was in the top results
        # Reward more chips for being in the top X
        if len(gt_pos_chip_list) > 0:
            # Use summation formula sum_i^n i = n(n+1)/2
            ret['gt_pos_chip'] = gt_pos_chip_list.min()
            _N = len(gt_pos_chip_list)
            _SUM_DENOM = float(_N * (_N + 1)) / 2.0
            ret['chip_error'] = float(gt_pos_chip_list.sum())/_SUM_DENOM
        # Calculate Precision / Recall (depends on the # threshold/max_results)
        ground_truth_cxs = np.setdiff1d(np.array(nm.nx2_cx_list[qnx]), np.array([qcx]))
        true_positives  = top_cx[gt_pos_chip_list-1]
        false_positives = np.setdiff1d(top_cx, true_positives)
        false_negatives = np.setdiff1d(ground_truth_cxs, top_cx)

        nTP = float(len(true_positives)) # Correct result
        nFP = float(len(false_positives)) # Unexpected result
        nFN = float(len(false_negatives)) # Missing result
        #nTN = float( # Correct absence of result

        ret['chip_precision'] = nTP / (nTP + nFP)
        ret['chip_recall']    = nTP / (nTP + nFN)
        #ret['true_negative_rate'] = nTN / (nTN + nFP)
        #ret['accuracy'] = (nTP + nFP) / (nTP + nTN + nFP + nFN)
        # ----
        # Score Top Names
        (top_nx, _) = res.nxcx_sort()
        gt_pos_name_list = (1+pylab.find(qnid == nm.nx2_nid[top_nx]))
        # If a correct name was in the top results
        if len(gt_pos_name_list) > 0: 
            ret['gt_pos_name'] = gt_pos_name_list.min() 
            # N should always be 1
            _N = len(gt_pos_name_list)
            _SUM_DENOM = float(_N * (_N + 1)) / 2.0
            ret['name_error'] = float(gt_pos_name_list.sum())/_SUM_DENOM
        # ---- 
        # RETURN RESULTS
        return ret


    def get_name_consistency_report_str(em):
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
        cx2_error_chip = -np.ones(len(em.cx2_res))
        cx2_error_name = -np.ones(len(em.cx2_res))
        # gt hist shows how often a chip is at rank X
        gt_hist_chip = np.zeros(len(cm.get_valid_cxs())+2) # add 2 because we arent using 0
        gt_hist_name = np.zeros(len(nm.get_valid_nxs())+2) # and 1 at the end for overflow
        for res in em.cx2_res:
            if res == []: continue
            qcx = res.rr.qcx
            nm_consist = em.res2_name_consistency(res)
            # Get the qualitative 'badness' of the results
            cx2_error_chip[qcx] = nm_consist['chip_error']
            cx2_error_name[qcx] = nm_consist['name_error']
            # Get the quantitative rankings of best results
            gt_hist_chip[nm_consist['gt_pos_chip']] += 1
            gt_hist_name[nm_consist['gt_pos_name']] += 1

        worst_cids = zip(cx2_error_name, range(len(cx2_error_name)))
        worst_nids = zip(cx2_error_chip, range(len(cx2_error_chip)))
        worst_cids.sort()
        worst_nids.sort()

        nid2_badness = -np.ones(len(nm.nid2_nx))

        report_str += 'WORST QUERY RESULTS:'+'\n'
        report_str += 'GT C AVE-RANK  | QCID - GT_CIDS, GT_CIDS_RANK'+'\n'
        for score, cx in worst_cids:
            if score < 0: continue        
            cid = cm.cx2_cid[cx]
            nid = cm.cx2_nid(cx)
            if nid2_badness[nid] == -1:
                nid2_badness[nid] = 0
            nid2_badness[nid] += score
            other_cxs  = np.setdiff1d(cm.cx2_other_cxs([cx])[0], [cx])
            top_cx = em.cx2_res[cx].cx_sort()
            other_rank = []
            for ocx in other_cxs:
                to_append = pylab.find(top_cx == ocx)+1
                other_rank.extend(to_append.tolist())
            other_cids = cm.cx2_cid[other_cxs]
            report_str += '%14.3f | %4d - %s' % (score, cid, str(zip(other_cids,
                                                                     np.array(other_rank,
                                                                              dtype=np.int32))))+'\n'


        report_str += '\nGT N RANK      | QNID | QCID - GT_CIDS'+'\n'
        for score, cx in worst_nids:
            if score < 0: continue        
            cid = cm.cx2_cid[cx]
            nid = cm.cx2_nid(cx)
            other_cids = np.setdiff1d(cm.cx2_cid[cm.cx2_other_cxs([cx])[0]], cid)
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



    # Old 
    def display_matching_results(em, fname=None):
        ''' displays results of a matching experiment'''
        cm, iom = em.hs.get_managers('cm','iom')
        fpath = iom.get_temp_fpath(fname)
        txt_match_fpath = iom.get_temp_fpath(fpath)
        num_show = 4
        cid_list = [0]*num_show
        titles = [None]*num_show
        import os
        try:
            os.makedirs(iom.get_temp_fpath('imgres'))
        except Exception:
            pass
        with open(txt_match_fpath, 'r') as file:
            file = open(txt_match_fpath, 'r')
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
