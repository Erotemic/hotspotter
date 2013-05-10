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


        # ---
    #def draw_graph(dm, G):
        #import networkx
        ##fig = dm.get_current_figure()
        #fig = figure(9001)
        #fig.clf()
        #ax = fig.gca()
        #pos = networkx.spring_layout(G, dim=2, scale=1,iterations=100) 
        ##pos = networkx.spectral_layout(G)
        ##pos = networkx.circular_layout(G)
        #node_labels=dict([(id,d['qnid']) for id,d in G.nodes(data=True)])
        ##networkx.draw_networkx(G,pos,ax=ax)
        #colormap = dm.hs.prefs['colormap']
        #cmap = get_cmap(colormap)
        #tot_num = dm.hs.nm.num_n+2
        #for cid in pos.keys():
            #cx = dm.hs.cm.cid2_cx[cid]
            #nid = dm.hs.cm.cx2_nid(cx)
            #color = cmap(float(nid)/tot_num)
            ##print color
            #networkx.draw_networkx_nodes(G,pos, nodelist=[cid], node_color=color, node_size=1000)
            #pass
        ##networkx.draw_networkx_nodes(G,pos)
        ##networkx.draw_networkx_labels(pos,node_labels)
        #networkx.draw_networkx_edges(G,pos,alpha=.5)
        #networkx.draw_networkx_labels(G,pos)
            
        #labels=networkx.draw_networkx_labels(G,pos=pos)
        #edge_labels=dict([((u,v,),'%.1f' % d['weight']) for u,v,d in G.edges(data=True)])
        #edge_labels = {}
        #networkx.draw_networkx_edge_labels(G,pos,edge_labels,alpha=0.5)
        #trans = ax.transData.transform
        #trans2 = fig.transFigure.inverted().transform
        #for node in G.nodes():
        #    x,y = pos[node]
