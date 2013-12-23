
def plot_cx3(hs, cx):
    ax = df2.gca()
    rchip = hs.get_chip(cx)
    ax.imshow(rchip, interpolation='nearest')
    df2.plt.set_cmap('gray')
    df2.set_ticks([],[])
    gname = hs.cx2_gname(cx)
    cid = hs.tables.cx2_cid[cx]
    ax.set_xlabel(gname)
    ax.set_title(hs.cxstr(cx))

def cx_info(allres, cx, SV=True):
    hs = allres.hs
    res = allres.qcx2_res[cx]
    print_top_res_scores(hs, res, view_top=10)
    #gt_cxs = hs.get_other_cxs(cx)
    gt_cxs = hs.get_other_indexed_cxs(cx)
    print('[viz] Ground truth '+hs.cx_liststr(gt_cxs))
    print('[viz] num groundtruth = '+str(len(gt_cxs)))
    print_top_res_scores(hs, res, view_top=10, SV=True)

def print_top_res_scores(hs, res, view_top=10, SV=True):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    lbl = ['(assigned)', '(assigned+V)'][SV]
    cx2_nx     = hs.tables.cx2_nx
    nx2_name   = hs.tables.nx2_name
    qnx        = cx2_nx[qcx]
    #other_cx   = hs.get_other_cxs(qcx)
    other_cx   = hs.get_other_indexed_cxs(qcx)
    top_cx     = cx2_score.argsort()[::-1]
    top_scores = cx2_score[top_cx]
    top_nx     = cx2_nx[top_cx]
    view_top   = min(len(top_scores), np.uint32(view_top))
    print('---------------------------------------')
    print('[viz]Inspecting matches of q%s name=%s' % (hs.cxstr(qcx), hs.cx2_name(qcx)))
    print('[viz] * Matched against %d other chips' % len(cx2_score))
    print('[viz] * Ground truth chip indexes:\n   gt_%s' % hs.cx_liststr(other_cx))
    print('[viz]The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('[viz]--> %s, score=%6.2f' % (hs.cxstr(cx, 4), score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        if tcx == qcx: continue
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print('[viz]'+_mark+' %s, score=%6.2f' % (hs.cxstr(cx), tscore))
    print('---------------------------------------')
    print('---------------------------------------')

def plot_cx(allres, cx, style='kpts', subdir=None, annotations=True, title_aug=''):
    hs    = allres.hs
    qcx2_res = allres.qcx2_res
    res = qcx2_res[cx]
    plot_cx2(hs, res, style=style, subdir=subdir, annotations=annotations, title_aug=title_aug)

def plot_cx2(hs, res, style='kpts', subdir=None, annotations=True, title_aug=''):
    #cx_info(allres, cx)
    cx = res.qcx
    raise Exception("fix no rr2")
    #title_suffix = rr2.get_title_suffix()
    if 'kpts' == style:
        subdir = 'plot_cx' if subdir is None else subdir
        rchip = hs.get_chip(cx)
        kpts  = hs.get_kpts(cx)
        print('[viz] Plotting'+title)
        df2.imshow(rchip, fignum=FIGNUM, title=title, doclf=True)
        df2.draw_kpts2(kpts)
    if 'gt_matches' == style:
        subdir = 'gt_matches' if subdir is None else subdir
        show_gt_matches(hs, res, fignum=FIGNUM)
    if 'top5' == style:
        subdir = 'top5' if subdir is None else subdir
        show_topN_matches(hs, res, N=5, fignum=FIGNUM)
    if 'analysis' == style:
        subdir = 'analysis' if subdir is None else subdir
        show_match_analysis(hs, res, N=5, fignum=FIGNUM,
                            annotations=annotations, figtitle=title_aug)
    subdir += title_suffix
    __dump_or_browse(hs, subdir)




# SPLIT PLOTS

def _show_res(hs, res, figtitle='', max_nCols=5, topN_cxs=None, gt_cxs=None,
              show_query=False, all_kpts=False, annote=True, query_cfg=None,
              split_plots=False, interact=True, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #printDBG('[viz._show_res()] %s ' % helpers.printableVal(locals()))
    fnum = kwargs.pop('fnum', 3)
    #print('========================')
    #print('[viz] Show chip matches:')
    if topN_cxs is None:
        topN_cxs = []
    if gt_cxs is None:
        gt_cxs = []
    qcx = res.qcx
    all_gts = hs.get_other_indexed_cxs(qcx)
    #print('[viz._show_res()]----------------')
    print('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % (len(topN_cxs),
                                                            len(gt_cxs),
                                                            len(all_gts)))
    #printDBG('[viz._show_res()] * max_nCols=%r' % (max_nCols,))
    #printDBG('[viz._show_res()] * show_query=%r' % (show_query,))
    ranked_cxs = res.topN_cxs(hs, N='all')
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts = nQuerySubplts + (0 if gt_cxs is None else len(gt_cxs))
    nTopNSubplts  = 0 if topN_cxs is None else len(topN_cxs)
    nTopNCols = min(max_nCols, nTopNSubplts)
    nGTCols   = min(max_nCols, nGtSubplts)
    if not split_plots:
        nGTCols = max(nGTCols, nTopNCols)
        nTopNCols = nGTCols
    nGtRows   = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells = nGtRows * nGTCols
    if split_plots:
        nRows = nGtRows
    else:
        nRows = nTopNRows + nGtRows
    # Helper function for drawing matches to one cx

    def _show_matches_fn(cx, orank, pnum):
        'helper for viz._show_res'
        aug = 'rank=%r\n' % orank
        #printDBG('[viz._show_res()] plotting: %r'  % (pnum,))
        kwshow  = dict(draw_ell=annote, draw_pts=annote, draw_lines=annote,
                       ell_alpha=.5, all_kpts=all_kpts, **kwargs)
        show_matches_annote_res(res, hs, cx, title_aug=aug, fnum=fnum, pnum=pnum, **kwshow)

    def _show_query_fn(plotx_shift, rowcols):
        'helper for viz._show_res'
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #printDBG('[viz._show_res()] Plotting Query: pnum=%r' % (pnum,))
        show_chip(hs, res=res, pnum=pnum, draw_kpts=annote, prefix='q', fnum=fnum)

    # Helper to draw many cxs
    def _plot_matches_cxs(cx_list, plotx_shift, rowcols):
        'helper for viz._show_res'
        #printDBG('[viz._show_res()] Plotting Chips %s:' % hs.cidstr(cx_list))
        if cx_list is None:
            return
        for ox, cx in enumerate(cx_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cxs == cx)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(cx, orank, pnum)

    #query_uid = res.query_uid
    #query_uid = re.sub(r'_trainID\([0-9]*,........\)', '', query_uid)
    #query_uid = re.sub(r'_indxID\([0-9]*,........\)', '', query_uid)
    #query_uid = re.sub(r'_dcxs\(........\)', '', query_uid)
    #print('[viz._show_res()] fnum=%r' % fnum)

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), doclf=True)
    fig.clf()
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols))
    # Plot TopN in a new figure
    if split_plots:
        #df2.set_figtitle(figtitle + 'GT', query_uid)
        nRows = nTopNRows
        fig = df2.figure(fnum=fnum + 9000)
        fig.clf()
        df2.plt.subplot(nRows, nTopNCols, 1)
        shift_topN = 0
    else:
        shift_topN = nGtCells
    _plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    if split_plots:
        pass
        #df2.set_figtitle(figtitle + 'topN', query_uid)
    else:
        pass
        #df2.set_figtitle(figtitle, query_uid)
        df2.set_figtitle(figtitle)

    if interact:
        #printDBG('[viz._show_res()] starting interaction')
        # Create
        def _on_res_click(event):
            'result interaction mpl event callback slot'
            print('[viz] clicked result')
            if event.xdata is None:
                return
            _show_res(hs, res, figtitle=figtitle, max_nCols=max_nCols, topN_cxs=topN_cxs,
                      gt_cxs=gt_cxs, show_query=show_query, all_kpts=all_kpts,
                      annote=not annote, split_plots=split_plots,
                      interact=interact, **kwargs)
            fig.canvas.draw()

        df2.disconnect_callback(fig, 'button_press_event')
        if interact:
            df2.connect_callback(fig, 'button_press_event', _on_res_click)
    #printDBG('[viz._show_res()] Finished')
    return fig



# USE LAB
    USE_LAB = False  # True  # False

    import tools
    from skimage import color
    if USE_LAB:
        isInt = tools.is_int(rchip2)
        rchip2_blendA = np.zeros((h2, w2, 3), dtype=rchip2.dtype)
        rchip2_blendH = np.zeros((h2, w2, 3), dtype=rchip2.dtype)
        rchip2_blendA = np.rollaxis(rchip2_blendA, 2)
        rchip2_blendH = np.rollaxis(rchip2_blendH, 2)
        #rchip2_blendA[0] = (rchip2 / 2) + (rchip1_At / 2)
        #rchip2_blendH[0] = (rchip2 / 2) + (rchip1_Ht / 2)
        #rchip2_blendA[0] /= 1 + (122 * isInt)
        #rchip2_blendH[0] /= 1 + (122 * isInt)
        rchip2_blendA[0] += 255
        rchip2_blendH[0] += 255
        rchip2_blendA[1] = rchip2
        rchip2_blendH[1] = rchip2
        rchip2_blendA[2] = rchip1_At
        rchip2_blendH[2] = rchip1_Ht
        rchip2_blendA = np.rollaxis(np.rollaxis(rchip2_blendA, 2), 2)
        rchip2_blendH = np.rollaxis(np.rollaxis(rchip2_blendH, 2), 2)
        print('unchanged stats')
        print(helpers.printable_mystats(rchip2_blendH.flatten()))
        print(helpers.printable_mystats(rchip2_blendA.flatten()))
        if isInt:
            print('is int')
            rchip2_blendA = np.array(rchip2_blendA, dtype=float)
            rchip2_blendH = np.array(rchip2_blendH, dtype=float)
        else:
            print('is float')
        print('div stats')
        print(helpers.printable_mystats(rchip2_blendH.flatten()))
        print(helpers.printable_mystats(rchip2_blendA.flatten()))
        rchip2_blendA = color.lab2rgb(rchip2_blendA)
        rchip2_blendH = color.lab2rgb(rchip2_blendH)
        if isInt:
            print('is int')
            rchip2_blendA = np.array(np.round(rchip2_blendA * 255), dtype=np.uint8)
            rchip2_blendH = np.array(np.round(rchip2_blendH * 255), dtype=np.uint8)
        print('changed stats')
        print(helpers.printable_mystats(rchip2_blendH.flatten()))
        print(helpers.printable_mystats(rchip2_blendA.flatten()))
