    # --- 
    #TODO This is so antiquated
    #def show_matching_graph(dm, computed_cx2_rr=None):
        #hs = dm.hs
        #cm = hs.cm
        #dm = hs.dm
        #try: 
            #computed_cx2_rr
        #except NameError:
            #computed_cx2_rr = None
        #if computed_cx2_rr == None:
            #computed_cx2_rr = vm.batch_query()
        #import networkx
        #G = networkx.DiGraph()
        #for rr in computed_cx2_rr:
            #if rr == []: continue
            #res = QueryResult(hs,rr)
            #qcid  = res.qcid
            #qnid  = res.qnid
            #G.add_node(qcid,qcid=qcid,qnid=qnid)
            #rr = res.rr
            #for (tscore,tcx) in zip(rr.cx2_cscore, range(len(rr.cx2_cscore))):
                #tcid = cm.cx2_cid[tcx]
                #if tscore > 0:
                    #G.add_edge(qcid, tcid, weight=tscore**2-tscore)
        #dm.draw_graph(G)
        ##hs.dm.end_draw()
        #pass
    # --- 
