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
        #helpers.vd(hs.dirs.result_dir)
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

