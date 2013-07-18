    # Skip 2 places do do symetrical matching
    matching_pairs_list = []
    symdid_set = set([]) # keeps track of symetric matches already computed



# cs prob

    withindb_chipscore_tup_list = []
    crossdb_chipscore_tup_list = []
            if hsA is hsB: 
                withindb_chipscore_tup_list.append((chipscore_data, ischipscore_TP))
            else:
                crossdb_chipscore_tup_list.append((chipscore_data, ischipscore_TP))

        
        if __FEATSCORE_STATISTICS__:
            print('  * Visualizing feature score statistics')
            # Visualize the individual feature match statistics
            output_dir = join_mkdir(result_dir, 'featmatch')
            (TP_inlier_score_list,
             TP_outlier_score_list,
             TP_inlier_scale_pairs,
             TP_outlier_scale_pairs,
             FP_inlier_score_list,
             FP_outlier_score_list,
             FP_inlier_scale_pairs,
             FP_outlier_scale_pairs) = get_featmatch_stats(hsA, hsB, count2rr_AB, i)
            fig_sd1, fig_fs1 = viz_featmatch_stats(outlier_scale_pairs, inlier_scale_pairs)
            fig_sd1.savefig(join(output_dir, results_name+'-scalediff.png'), format='png')
            fig_fs1.savefig(join(output_dir, results_name+'-fmatchscore.png'), format='png')
            

        if __SYMETRIC_MATCHINGS__:
            # Visualize chips which symetrically.
            # Do not symetric match twice 
            if (hsA, hsB) in symdid_set: 
                print('  * Already computed symetric matchings')
                continue
            else: 
                symdid_set.add((hsA, hsB))
                symdid_set.add((hsB, hsA))
            output_dir = join_mkdir(result_dir, 'symetric_matches')
            # Find the symetric index
            symx = symx_list[count]
            count2rr_BA = count2rr_list[symx]
            print('  * Visualizing symetric matchings')
            matching_pairs = get_symetric_matchings(hsA, hsB, count2rr_AB, count2rr_BA)
            viz_symetric_matchings(matching_pairs, results_name, output_dir)




        # endfor
    print('Aggregate Visualizations: ')
    if __AGGREGATE_CHIPSCORES__:
        output_dir  = join_mkdir(result_dir, 'chipscore_frequencies')
        for holdon in [False, True]:
            fig = figure(0, figsize=__FIGSIZE__)
            fig.clf()
            within_lbl = ['within-db','within-db-sansgt'][__SANS_GT__]
            cross_lbl = 'cross-db'
            agg_chipscore_titlestr = \
                    'Probability desnity of chip scores \n' + \
                    within_lbl+'-databases vs cross-databases'
            if __METHOD_AND_K_IN_TITLE__:
                agg_chipscore_titlestr += '\nscored with: '+__METHOD__+' k='+str(__K__)

            title(agg_chipscore_titlestr)
            fig.canvas.set_window_title(agg_chipscore_titlestr)
            aggchipscore_color = [1,0,0]
            for expt_type, chipscore_tup_list in zip((within_lbl, cross_lbl),\
                                                     (withindb_chipscore_tup_list,\
                                                      crossdb_chipscore_tup_list)):
                print ('  * Visualizing '+expt_type+' Aggregate Chipscores')
                expt_type_full     = expt_type.replace('db', 'database')
                aggchipscore_fname = join(output_dir, expt_type+'-aggchipscore')
                if len(chipscore_tup_list) > 0:
                    chipscore_data_all, ischipscore_TP_all =  [list(t) for t in \
                                                                zip(*chipscore_tup_list)]
                    aggchipscore_data = np.vstack(chipscore_data_all)
                    aggischipscore_TP = np.vstack(ischipscore_TP_all)
                    bigplot_chipscores(expt_type_full+' experiments',
                                    aggchipscore_data,
                                    aggischipscore_TP,
                                    aggchipscore_fname,
                                    releaseaxis=True,
                                    color=aggchipscore_color,
                                    releasetitle=(not holdon),
                                    labelaug=expt_type+' ',
                                    sameplot=True, 
                                    holdon=holdon)
                aggchipscore_color = [0,0,1]
            if holdon == True:
                aggchipscore_fname = join(output_dir, within_lbl+'-'+cross_lbl+'aggchipscore')
                legend()
                safe_savefig(fig, aggchipscore_fname+'.png')
    print("Vizualizations Complete")





    
# Visualization of chips which match symetrically (in top x results)
def viz_symetric_matchings(hsA, hsB, matching_pairs, results_name, output_dir='symetric_matches'):
    print('  * Visualizing '+str(len(matching_pairs))+' matching pairs')
    for cx, cx2, match_pos, match_pos1, res1, res2 in matching_pairs:
        for res, suffix in zip((res1,res2), ('AB','BA')):
            res.visualize()
            fignum = 0
            fig = figure(num=fignum, figsize=__FIGSIZE__)
            #fig.show()
            fig.canvas.set_window_title('Symetric Matching: '+str(cx)+' '+str(cx2))
            fig_fname = results_name+\
                    '__symmpos_'+str(match_pos)+'_'+str(match_pos1)+\
                    '__cx_'+str(cx)+'_'+str(cx2)+\
                    suffix+\
                    '.png'
            fig_fpath = realpath(join(output_dir, fig_fname))
            print('      * saving to '+fig_fpath)
            safe_savefig(fig, fig_fpath)
            fig.clf()




# Visualization of feature score probability
def viz_fmatch_score(results_name, inlier_score_list, outlier_score_list, fignum=0):
    inlier_scores  = np.array(inlier_score_list)
    outlier_scores = np.array(outlier_score_list)
    # Set up axes and labels: fscores
    fig_scorediff = figure(num=fignum, figsize=__FIGSIZE__)
    fig_scorediff.clf()
    title_str = 'Frequency of feature scores \n'+results_name
    if __METHOD_AND_K_IN_TITLE__:
        title_str += '\nscored with: '+__METHOD__+' k='+str(__K__)
    xlabel('feature score ('+hsB.am.algo_prefs.query.method+')')
    ylabel('frequency')
    title(title_str)
    inlier_args  = {'label':'P(fscore | inlier)',  'color':[0,0,1]}
    outlier_args = {'label':'P(fscore | outlier)', 'color':[1,0,0]}
    # histogram 
    hist(inlier_scores,  normed=1, alpha=.3, bins=100, **inlier_args)
    hist(outlier_scores, normed=1, alpha=.3, bins=100, **outlier_args)
    # pdf
    fx_extent = (0, max(inlier_scores.max(), outlier_scores.max()))
    fs_domain = np.linspace(fx_extent[0], fx_extent[1], 100)
    inscore_pdf = gaussian_kde(inlier_scores)
    outscore_pdf = gaussian_kde(outlier_scores)
    plot(fs_domain, outscore_pdf(fs_domain), **outlier_args) 
    plot(fs_domain, inscore_pdf(fs_domain),  **inlier_args) 

    legend()
    #fig_scorediff.show()
    return fig_scorediff

# Visualization of scale difference probability
def viz_fmatch_scalediff(hsA, hsB, outlier_scale_pairs, inlier_scale_pairs, fignum=0):
    out_scales = np.array(outlier_scale_pairs)
    in_scales = np.array(inlier_scale_pairs)
    out_scale_diff = np.abs(out_scales[:,0] - out_scales[:,1])
    in_scale_diff = np.abs(in_scales[:,0] - in_scales[:,1])
    # Remove some extreme data
    in_scale_diff.sort() 
    out_scale_diff.sort() 
    subset_in  = in_scale_diff[0:int(len(in_scale_diff)*.88)]
    subset_out = out_scale_diff[0:int(len(out_scale_diff)*.88)]
    # Set up axes and labels: scalediff
    fig_scalediff = figure(num=fignum, figsize=__FIGSIZE__)
    fig_scalediff.clf()
    title_str = 'Frequency of feature scale differences (omitted largest 12%) \n' + \
        'queries from: '+hsA.get_dbid() + '\n' + \
        'results from: '+hsB.get_dbid()
    if __METHOD_AND_K_IN_TITLE__:
        title_str += '\nscored with: '+__METHOD__+' k='+str(__K__)
    xlabel('scale difference')
    ylabel('frequency')
    title(title_str)
    fig_scalediff.canvas.set_window_title(title_str)
    inlier_args  = {'label':'P( scale_diff | inlier )',  'color':[0,0,1]}
    outlier_args = {'label':'P( scale_diff | outlier )', 'color':[1,0,0]}
    # histogram 
    hist(subset_in, normed=1, alpha=.3, bins=100,  **inlier_args)
    hist(subset_out, normed=1, alpha=.3, bins=100, **outlier_args)
    # pdf
    sd_extent = (0, max(subset_in.max(), subset_out.max()))
    sd_domain = np.linspace(sd_extent[0], sd_extent[1], 100)
    subset_out_pdf = gaussian_kde(subset_out)
    subset_in_pdf = gaussian_kde(subset_in)
    plot(sd_domain, subset_in_pdf(sd_domain), **inlier_args) 
    plot(sd_domain, subset_out_pdf(sd_domain), **outlier_args) 
    
    legend()
    #fig_scalediff.show()
    return fig_scalediff




def get_symetric_matchings(hsA, hsB, count2rr_AB, count2rr_BA):
    'returns database, cx, database cx'
    import numpy as np
    sym_match_thresh = 5

    matching_pairs = []
    valid_cxsB = hsB.cm.get_valid_cxs()
    lop_thresh = 10
    for count in xrange(len(count2rr_AB)):
        rr = count2rr_AB[count]
        cx = rr.qcx
        res = QueryResult(hsB, rr, hsA)
        top_cxs = res.top_cx()
        top_scores = res.scores()[top_cxs]
        level_of_promise = len(top_scores) > 0 and top_scores[0]
        if level_of_promise > lop_thresh:
            lop_thresh = lop_thresh + (0.2 * lop_thresh)
            print('    * Checking dbA cx='+str(cx)+' \n'+\
                  '      top_cxs='+str(top_cxs)+'\n'+\
                  '      top_scores='+str(top_scores))
        match_pos1 = -1
        for tcx, score in zip(top_cxs, top_scores):
            match_pos1 += 1
            count = (valid_cxsB == tcx).nonzero()[0]
            rr2  = count2rr_BA[count]
            res2 = QueryResult(hsA, rr2, hsB)
            top_cxs2    = res2.top_cx()
            top_scores2 = res2.scores()[top_cxs2]
            if level_of_promise > lop_thresh:
                print('      * topcxs2 = '+str(top_cxs2))
                print('      * top_scores2 = '+str(top_scores2))
            # Check if this pair has eachother in their top 5 results
            match_pos_arr = (top_cxs2 == cx).nonzero()[0]
            if len(match_pos_arr) == 0: continue
            match_pos = match_pos_arr[0]
            print('  * Symetric Match: '+str(cx)+' '+str(tcx)+'   match_pos='+str(match_pos)+', '+str(match_pos1))
            
            matching_pairs.append((cx, tcx, match_pos, match_pos1, res, res2))
    return matching_pairs

def get_featmatch_stats(hsA, hsB, count2rr_AB):
    num_queries = len(count2rr_AB)
    TP_inlier_score_list   = []
    TP_outlier_score_list  = []
    TP_inlier_scale_pairs  = []
    TP_outlier_scale_pairs = []
    # False positive in this case also encompases False Negative and unknown
    FP_inlier_score_list   = []
    FP_outlier_score_list  = []
    FP_inlier_scale_pairs  = []
    FP_outlier_scale_pairs = []
    # Get Data
    print('Aggregating featmatch info for  '+str(num_queries)+' queries')
    for count in xrange(num_queries):
        rr = count2rr_AB[count]
        res = QueryResult(hsA, rr, hsB)
        # Get the cxs which are ground truth
        gtcx_list = res.get_groundtruth_cxs()
        qcx = rr.qcx
        qname = hsA.cm.cx2_name(qcx)
        # Get query features
        qfpts, _ = hsA.cm.get_feats(qcx)
        for cx in xrange(len(rr.cx2_fm)):
            # Switch to whatever the correct list to append to is
            if not gtcx_list is None and cx in gtcx_list:
                inlier_score_list   = TP_inlier_score_list   
                outlier_score_list  = TP_outlier_score_list  
                inlier_scale_pairs  = TP_inlier_scale_pairs  
                outlier_scale_pairs = TP_outlier_scale_pairs 
            else:
                inlier_score_list   = FP_inlier_score_list   
                outlier_score_list  = FP_outlier_score_list  
                inlier_scale_pairs  = FP_inlier_scale_pairs  
                outlier_scale_pairs = FP_outlier_scale_pairs 

            # Get feature matching indexes and scores
            feat_matches = rr.cx2_fm[cx]
            feat_scores_SC  = rr.cx2_fs[cx]
            feat_scores_all = rr.cx2_fs_[cx]
            nx = []
            name = hsB.cm.cx2_name(cx)
            # continue if no feature matches
            if len(feat_matches) == 0: continue
            # Get database features
            fpts, _ = hsB.cm.get_feats(cx)
            # Separate into inliers / outliers
            outliers = (feat_scores_SC == -1)
            inliers = True - outliers
            # Get info about matching scores
            outlier_scores = feat_scores_all[outliers]
            inlier_scores  = feat_scores_all[inliers]
            # Append score info
            inlier_score_list.extend(inlier_scores.tolist())
            outlier_score_list.extend(outlier_scores.tolist())
            # Get info about matching keypoint shape
            inlier_matches  = feat_matches[inliers]
            outlier_matches = feat_matches[outliers]
            inlier_qfpts  = qfpts[inlier_matches[:,0]]
            outlier_qfpts = qfpts[outlier_matches[:,0]]
            inlier_fpts   =  fpts[inlier_matches[:,1]]
            outlier_fpts  =  fpts[outlier_matches[:,1]]
            # Get the scales of matching keypoints as their sqrt(1/determinant)
            aQI,_,dQI = inlier_qfpts[:,2:5].transpose()
            aDI,_,dDI = inlier_fpts[:,2:5].transpose()
            aQO,_,dQO = outlier_qfpts[:,2:5].transpose()
            aDO,_,dDO = outlier_fpts[:,2:5].transpose()
            inlier_scalesA  = np.sqrt(1/np.multiply(aQI,dQI))
            inlier_scalesB  = np.sqrt(1/np.multiply(aDI,dDI))
            outlier_scalesA = np.sqrt(1/np.multiply(aQO,dQO))
            outlier_scalesB = np.sqrt(1/np.multiply(aDO,dDO))
            # Append to end of array
            outlier_scale_pairs.extend(zip(outlier_scalesA, outlier_scalesB))
            inlier_scale_pairs.extend(zip(inlier_scalesA, inlier_scalesB))
    return (TP_inlier_score_list,
            TP_outlier_score_list,
            TP_inlier_scale_pairs,
            TP_outlier_scale_pairs,
            FP_inlier_score_list,
            FP_outlier_score_list,
            FP_inlier_scale_pairs,
            FP_outlier_scale_pairs)




def high_score_matchings():
    'Visualizes each query against its top matches'
    pass


def scoring_metric_comparisons():
    'Plots a histogram of match scores of differnt types'
    pass


def spatial_consistent_match_comparisons():
    'Plots a histogram of spatially consistent match scores vs inconsistent'
    pass



    if __CHIPSCORE_PROBAILITIES__:
        combo_legend_size = 18
        print('Plotting cross-database chip score frequencies')
        # PLOT CROSS-DATABASE PLOTS ON THE SAME GRAPH
        combocrossdb_fname = join(_CHIPSCORE_DIR_, 'combo-crossdb-chipscore')
        fig = figure(0, figsize=__FIGSIZE__)
        fig.clf()
        title('Frequency of cross-database chip scores')
        tmp_count = 0.0
        tmp_total = len(dbvs_list)+1
        for i in range(len(dbvs_list)):
            hsA, hsB = dbvs_list[i]
            # ONLY CROSS DATABASE
            if hsA is hsB: continue
            results_name = get_results_name(hsA, hsB) 
            print('  * combining: '+results_name)
            count2rr_AB = count2rr_list[i]
            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)
            bigplot_chipscores(results_name,
                               chipscore_data,
                               ischipscore_TP,
                               'NA-fname',
                               labelaug=results_name+' ',
                               releaseaxis=True, 
                               releasetitle=False, 
                               color=get_cmap('Set1')(tmp_count/tmp_total),
                               plotTPFP=False,
                               sameplot=True,
                               holdon=True)
            tmp_count += 1
        legend(prop={'size':combo_legend_size})
        #fig.show()
        # TRUNCATION HACK
        ax = fig.get_axes()[0]
        ax.set_xlim(0,50)
        safe_savefig(fig, combocrossdb_fname+'.png')

        # PLOT WITHIN-DATABASE TRUE POSITIVE PLOTS ON THE SAME GRAPH
        print('Plotting '+within_label+' chip score frequencies')
        
        combocrossdb_fname = join(_CHIPSCORE_DIR_, 'combo-'+within_lbl+'-chipscore')


        fig0 = figure(0, figsize=__FIGSIZE__)
        fig0.clf()
        title('Frequency of '+within_lbl+' chip scores')

        fig1 = figure(1, figsize=__FIGSIZE__)
        fig1.clf()
        title('Frequency of '+within_lbl+' TRUE positive chip scores')

        fig2 = figure(2, figsize=__FIGSIZE__)
        fig2.clf()
        title('Frequency of '+within_lbl+' FALSE positive chip scores')

        tmp_count = 0.0
        tmp_total = len(dbvs_list)+1
        for i in range(len(dbvs_list)):
            hsA, hsB = dbvs_list[i]
            # ONLY WITHIN DATABASE
            if not hsA is hsB: continue
            results_name = get_results_name(hsA, hsB) 
            print('  * combining: '+results_name)
            count2rr_AB = count2rr_list[i]
            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)
            bigplot_chipscores(results_name,
                               chipscore_data,
                               ischipscore_TP,
                               'NA-fname',
                               labelaug=results_name+' ',
                               releaseaxis=True, 
                               releasetitle=False, 
                               color=get_cmap('Set1')(tmp_count/tmp_total),
                               plotTPFP=True,
                               sameplot=False,
                               holdon=True)
            tmp_count += 1
        # Save combo tp, fp 
        # TRUNCATION HACK
        fig0 = figure(0, figsize=__FIGSIZE__)
        ax = fig0.get_axes()[0]
        ax.set_xlim(10,200)
        legend(prop={'size':combo_legend_size})
        safe_savefig(fig0, combocrossdb_fname+'.png')

        # TRUNCATION HACK
        fig1 = figure(1, figsize=__FIGSIZE__)
        ax = fig1.get_axes()[0]
        ax.set_xlim(10,200)
        legend(prop={'size':combo_legend_size})
        safe_savefig(fig1, combocrossdb_fname+'-TP.png')

        # TRUNCATION HACK
        fig2 = figure(2, figsize=__FIGSIZE__)
        ax = fig2.get_axes()[0]
        ax.set_xlim(0,60)
        legend(prop={'size':combo_legend_size})
        safe_savefig(fig2, combocrossdb_fname+'-FP.png')

        #return 




# --- Big Plots ---

def bigplot_chipscores(results_name,
                       chipscore_data,
                       ischipscore_TP,
                       chipscore_fname,
                       sameplot=False,
                       holdon=False,
                       plotTPFP=False,
                       releasetitle=None,
                       releaseaxis=None,
                       color=None,
                       **kwargs):

    if holdon:
        kwargs['holdon'] = True
    if not color is None:
        kwargs['color'] = color

    kwargs['releaseaxis'] = True if releaseaxis is None else releaseaxis
    kwargs['releasetitle'] = True if releasetitle is None else releasetitle
    if sameplot: 
        fignum = 0
        fig = figure(0, figsize=__FIGSIZE__)
        if not holdon:
            fig.clf()
        kwargs['holdon'] = True
        kwargs['color'] = [0,0,1] if color is None else color
    else: 
        fignum=0

    fig_chipscore  = viz_chipscores(results_name,
                                    chipscore_data,
                                    ischipscore_TP,
                                    restype='',
                                    fignum=fignum, **kwargs)
    if plotTPFP: # DO TP / FP on self queries and not __SANS_GT__
        if sameplot: 
            kwargs['color'] = [0,1,0] if color is None else color
            kwargs['releaseaxis'] = False
            kwargs['releasetitle'] = False
        else: 
            fignum = 1
        print('\n\nplotting TRUE positives')
        fig_chipscoreTP = viz_chipscores(results_name,
                                         chipscore_data,
                                         ischipscore_TP,
                                         restype='TP',
                                         fignum=fignum,
                                         **kwargs)
        print('\n\nDONE plotting TRUE positives\n\n*****************\n')
        if sameplot: 
            kwargs['color'] = [1,0,0]  if color is None else color
        else: 
            fignum = 2
        print('plotting false positives')
        fig_chipscoreFP = viz_chipscores(results_name,
                                         chipscore_data,
                                         ischipscore_TP,
                                         restype='FP', 
                                         fignum=fignum,
                                         **kwargs)
    if sameplot:
        if not holdon:
            #fig.show()
            legend()
            safe_savefig(fig, chipscore_fname+'.png')
        return fig
    else:
        if not holdon:
            safe_savefig(fig_chipscore, chipscore_fname+'.png')
        if plotTPFP:
            if not holdon:
                safe_savefig(fig_chipscoreTP, chipscore_fname+'TP.png')
                safe_savefig(fig_chipscoreFP, chipscore_fname+'FP.png')
            return fig_chipscore, fig_chipscoreTP, fig_chipscoreFP
        else:
            return fig_chipscore


__METHOD_AND_K_IN_TITLE__ = False
if __METHOD_AND_K_IN_TITLE__:
    title_str += '\nscored with: '+__METHOD__+' k='+str(__K__)

        #plt.hist(scores, normed=1, range=(0,max_score), alpha=.1, log=True,
        #bins=np.logspace(0,3,100),  histtype='stepfilled', figure=fig)) # , bins=max_score/2 
       
               y_data = score_pdf(scores)+perb
        y_data = [(highpdf - lowpdf)/2. for _ in scores]

        histscale = np.logspace(0,3,100)
        histscale = histscale[histscale - 10 > 5] 
        histscale = [10, 20, 40, 60, 80, 100, 120, 160,
                    200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
                    1000]
