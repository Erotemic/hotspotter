from os.path import realpath, join
import os
# Global variables
BROWSE = True
DUMP = False
FIGNUM = 1

def plot_rank_stem(allres, orgres_type='true'):
    print('[viz] plotting rank stem')
    # Visualize rankings with the stem plot
    hs = allres.hs
    title = orgres_type + 'rankings stem plot\n' + allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    x_data = orgres.qcxs
    y_data = orgres.ranks
    df2.draw_stems(x_data, y_data)
    slice_num = int(np.ceil(np.log10(len(orgres.qcxs))))
    df2.set_xticks(hs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))
    __dump_or_browse(allres.hs, 'rankviz')


def plot_rank_histogram(allres, orgres_type):
    print('[viz] plotting %r rank histogram' % orgres_type)
    ranks = allres.__dict__[orgres_type].ranks
    label = 'P(rank | ' + orgres_type + ' match)'
    title = orgres_type + ' match rankings histogram\n' + allres.title_suffix
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=label)  # FIXME
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'rankviz')


def plot_score_pdf(allres, orgres_type, colorx=0.0, variation_truncate=False):
    print('[viz] plotting ' + orgres_type + ' score pdf')
    title  = orgres_type + ' match score frequencies\n' + allres.title_suffix
    scores = allres.__dict__[orgres_type].scores
    print('[viz] len(scores) = %r ' % (len(scores),))
    label  = 'P(score | %r)' % orgres_type
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    if variation_truncate:
        df2.variation_trunctate(scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'scoreviz')


def plot_score_matrix(allres):
    print('[viz] plotting score matrix')
    score_matrix = allres.score_matrix
    title = 'Score Matrix\n' + allres.title_suffix
    # Find inliers
    #inliers = helpers.find_std_inliers(score_matrix)
    #max_inlier = score_matrix[inliers].max()
    # Trunate above 255
    score_img = np.copy(score_matrix)
    score_img[score_img < 0] = 0
    score_img[score_img > 255] = 255
    #dim = 0
    #score_img = helpers.norm_zero_one(score_img, dim=dim)
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    df2.imshow(score_img, fnum=FIGNUM)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    __dump_or_browse(allres.hs, 'scoreviz')


# Dump logic
def __browse():
    print('[viz] Browsing Image')
    df2.show()


def __dump(hs, subdir):
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None:
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()


def save_if_requested(hs, subdir):
    if not hs.args.save_figures:
        return
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None:
        subdir = helpers.sanatize_fname2(subdir)
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()


def __dump_or_browse(hs, subdir=None):
    #fig = df2.plt.gcf()
    #fig.tight_layout()
    if BROWSE:
        __browse()
    if DUMP:
        __dump(hs, subdir)


def plot_tt_bt_tf_matches(hs, allres, qcx):
    #print('Visualizing result: ')
    #res.printme()
    res = allres.qcx2_res[qcx]
    ranks = (allres.top_true_qcx_arrays[0][qcx],
             allres.bot_true_qcx_arrays[0][qcx],
             allres.top_false_qcx_arrays[0][qcx])
    #scores = (allres.top_true_qcx_arrays[1][qcx],
             #allres.bot_true_qcx_arrays[1][qcx],
             #allres.top_false_qcx_arrays[1][qcx])
    cxs = (allres.top_true_qcx_arrays[2][qcx],
           allres.bot_true_qcx_arrays[2][qcx],
           allres.top_false_qcx_arrays[2][qcx])
    titles = ('best True rank=' + str(ranks[0]) + ' ',
              'worst True rank=' + str(ranks[1]) + ' ',
              'best False rank=' + str(ranks[2]) + ' ')
    df2.figure(fnum=1, pnum=231)
    res.plot_matches(res, hs, cxs[0], False, fnum=1, pnum=131, title_aug=titles[0])
    res.plot_matches(res, hs, cxs[1], False, fnum=1, pnum=132, title_aug=titles[1])
    res.plot_matches(res, hs, cxs[2], False, fnum=1, pnum=133, title_aug=titles[2])
    fig_title = 'fig q' + hs.cidstr(qcx) + ' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)


def dump_gt_matches(allres):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        res.show_gt_matches(hs, fnum=FIGNUM)
        __dump_or_browse(allres.hs, 'gt_matches' + allres.title_suffix)


def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fnum=FIGNUM, pnum=121)
        df2.show_matches3(res, hs, cx, SV=False, fnum=FIGNUM, pnum=121)
        df2.show_matches3(res, hs, cx, SV=True,  fnum=FIGNUM, pnum=122)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % (score, rank, query_gname,
                                                      result_gname)
        df2.set_figtitle(big_title)
        __dump_or_browse(allres.hs, orgres_type + '_matches' + allres.title_suffix)

