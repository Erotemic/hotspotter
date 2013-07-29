import numpy as np
from hotspotter.helpers import alloc_lists
from hotspotter.other.ConcretePrintable import Pref
from hotspotter.tpl.pyflann import FLANN
#from __future__ import print_function
#def flann_one_time(data_vecs, query_vecs, K, flann_args):
    #N = query_vecs.shape[0]
# TODO: Clean up vector dtype format. Probably too much casting going on
# Paramatarize this in AlgoManager
def approximate_kmeans(data, K=1e6, max_iters=1000, flann_pref=None):
    if flann_pref == None:
        flann_pref = Pref()
        flann_pref.algorithm  = Pref('kdtree')
        flann_pref.trees      = Pref(8)
        flann_pref.checks     = Pref(128)
    flann_args = flann_pref.to_dict()
    float_data = np.array(data, dtype=np.float32)
    N = float_data.shape[0]
    print('Approximately clustering %d data vectors into %d clusters' % (N, K))
    np.random.seed(seed=0) # For Reproducibility
    # Initialize to Random Cluster Centers
    centx  = np.random.choice(N, size=K, replace=False)
    cent   = np.copy(float_data[centx])
    assign = alloc_lists(K) # List for each cluster center with assigned indexes
    for iterx in xrange(0,max_iters):
        print "Iteration " + str(iterx)
        # Step 1: Find Nearest Neighbors
        flann       = FLANN()
        flann.build_index(data_vecs, **flann_args)
        (index_list, dist_list) = flann.nn_index(query_vecs, K, checks=flann_args['checks'])
        return (index_list, dist_list)
        datax2_centx, _ = flann_one_time(cent, float_data, 1, flann_args)
        # Step 2: Assign data to cluster centers
        datax_sort = datax2_centx.argsort()
        centx_sort = datax2_centx[datax_sort]
        # Efficiently Trace over sorted centers with two pointers. Take care
        # To include the last batch of datavecs with the same center_index
        converged = True
        prev_centx = -1
        _L = 0
        dbg_total_assigned = 0
        dbg_assigned_list = []
        for _R in xrange(N+1): #Loop over datapoints, going 1 past the end, and group them
            # data  =  0[  . . . . . . . . . . . . .]N
            # ptrs  =          L         R
            #                  |-   k  -|L       R
            #                            |- k+1 |L   R     
            #                                    |_K|       
            if _R == N or centx_sort[_L] != centx_sort[_R]: # We found a group
                centx = centx_sort[_L] # Assign this group cluster index: centx
                # SPECIAL CASE: ( akmeans might not assign everything )
                if centx - prev_centx > 1: #Check if a cluster got skipped
                    for skipx in xrange(prev_centx+1, centx):
                        print("    Skipping Index:" + str(skipx))
                        if len(assign[skipx]) != 0:
                            converged = False
                        assign[skipx] = []
                prev_centx = centx
                # Set Assignments
                num_members = np.float32(_R - _L)
                dbg_total_assigned += num_members
                centx_membx = datax_sort[_L:_R]
                #DBG CODE, keep track of data vectors you've assigned
                #print('    Assigning %d data vectors to center index: %d' % (num_members, centx) )
                #for x in centx_membx:
                    #dbg_assigned_list.append(x)
                #/DBGCODE
                if np.all(assign[centx] != centx_membx):
                    converged = False
                assign[centx] = centx_membx
                # Recompute Centers
                cent[centx] = float_data[centx_membx,:].sum(axis=0) / num_members
                _L = _R
        #print('    Did Assignment of %d centers' % prev_centx)
        #print('    Assigned %d datavectors in total' % dbg_total_assigned)
        # SPECIAL CASE: has to run at the end again
        if prev_centx < K: #Check if a cluster got skipped at the end
            for skipx in xrange(prev_centx+1, K):
                print('    Cluster Index %d was empty:' % skipx)
                if len(assign[skipx]) != 0:
                    converged = False
                assign[skipx] = []
        prev_centx = centx

        if converged: # Assignments have not changed
            print 'akmeans converged in '+str(iterx)+' iterations'
            break
    return cent, assign
