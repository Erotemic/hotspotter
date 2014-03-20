    #viz.DUMP = True
    #viz.BROWSE = False

    # Dump text results, the stem plot, and the failure cases
    #dump_text_results(allres)
    #viz.plot_rank_stem(allres, 'true')
    #for cx in greater5_cxs:
        #top5(cx)
        #gt_matches(cx)

    #viz.DUMP = False
    #viz.BROWSE = True
    #cx = greater5_cxs[0] if len(greater5_cxs) > 0 else 0
    #top5(cx)
    #matches(cx)

    # IPYTHON END

    #nonipython_exec = textwrap.dedent(r"""
    #help_ = textwrap.dedent(r'''
    #Enter a command.
        #q (or space) : quit
        #h            : help
        #cx [cx]    : shows a chip
    #''')
    #print(help_)
    #firstcmd = 'cx 0'
    #firstcmd = 'stem'
    #ans = None
    #viz.DUMP = False
    #while True:
        ## Read command or run the first one hardcoded in
        #ans = raw_input('>') if not ans is None else firstcmd
        #if ans == 'q' or ans == ' ':
            #break
        #if allres is None:
            #print('Loading hotspotter')
        #if ans == 'h':
            #print help_
        #elif re.match('cx [0-9][0-9]*', ans) or\
             #re.match('[0-9][0-9]*', ans):
            #cx = int(ans.replace('cx ',''))
            #selc(cx)
        #elif ans == 'stem':
            #viz.plot_rank_stem(allres, 'true')
        #else:
            #exec(ans)
        #df2.update()
    ##browse='--browse' in sys.argv
    ##stem='--stem' in sys.argv
    ##hist='--hist' in sys.argv
    ##pdf='--pdf'   in sys.argv
    #viz.DUMP = True
    #dump_all(allres)
    #if '--vrd' in sys.argv:
        #util.vd(hs.dirs.result_dir)
    ##dinspect(18)
    #print(allres)
    #exec(df2.present(wh=(900,600)))
    #viz.DUMP = False
    #""")



    #cx2_nx = hs.tables.cx2_nx
    # Build name-to-chips dict
    #nx2_cxs = {}
    #for cx, nx in enumerate(cx2_nx):
        #if not nx in nx2_cxs.keys():
            #nx2_cxs[nx] = []
        #nx2_cxs[nx].append(cx)
    #nx_list = nx2_cxs.keys()
    #nx_size = [len(nx2_cxs[nx]) for nx in nx_list]

    #cx_sorted = hs.nx2_cxs(nx_list)
    #for nx in iter(nx_sorted):
        #cxs = nx2_cxs[nx]
        #cx_sorted.extend(sorted(cxs))
    # get matrix data rows


def print_result_summaries_list(topnum=5):
    print('\n<(^_^<)\n')
    # Print out some summary of all results you have
    hs = ld2.HotSpotter()
    hs.load_tables(ld2.DEFAULT)
    result_file_list = os.listdir(hs.dirs.result_dir)

    sorted_rankres = []
    for result_fname in iter(result_file_list):
        if fnmatch.fnmatch(result_fname, 'rankres_str*.csv'):
            print(result_fname)
            with open(join(hs.dirs.result_dir, result_fname), 'r') as file:

                metaline = file.readline()
                toprint = metaline
                # skip 4 metalines
                [file.readline() for _ in xrange(4)]
                top5line = file.readline()
                top1line = file.readline()
                toprint += top5line + top1line
                line = read_until(file, '# NumData')
                num_data = int(line.replace('# NumData', ''))
                file.readline()  # header
                res_data_lines = [file.readline() for _ in xrange(num_data)]
                res_data_str = np.array([line.split(',') for line in res_data_lines])
                tt_scores = np.array(res_data_str[:, 5], dtype=np.float)
                bt_scores = np.array(res_data_str[:, 6], dtype=np.float)
                tf_scores = np.array(res_data_str[:, 7], dtype=np.float)

                tt_score_sum = sum([score for score in tt_scores if score > 0])
                bt_score_sum = sum([score for score in bt_scores if score > 0])
                tf_score_sum = sum([score for score in tf_scores if score > 0])

                toprint += ('tt_scores = %r; ' % tt_score_sum)
                toprint += ('bt_scores = %r; ' % bt_score_sum)
                toprint += ('tf_scores = %r; ' % tf_score_sum)
                if topnum == 5:
                    sorted_rankres.append(top5line + metaline)
                else:
                    sorted_rankres.append(top1line + metaline)
                print(toprint + '\n')

    print('\n(>^_^)>\n')

    sorted_mapscore = []
    for result_fname in iter(result_file_list):
        if fnmatch.fnmatch(result_fname, 'oxsty_map_csv*.csv'):
            print(result_fname)
            with open(join(hs.dirs.result_dir, result_fname), 'r') as file:
                metaline = file.readline()
                scoreline = file.readline()
                toprint = metaline + scoreline

                sorted_mapscore.append(scoreline + metaline)
                print(toprint)

    print('\n'.join(sorted(sorted_rankres)))
    print('\n'.join(sorted(sorted_mapscore)))

    print('\n^(^_^)^\n')

