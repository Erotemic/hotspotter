    
def __affine_inliers(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
    best_inliers = []
    best_Aff = None
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    inv2_m = inv_acd(acd2_m, det2_m)
    # The transform from kp1 to kp2 is given as:
    # A = inv(A2).dot(A1)
    Aff_list = dot_acd(inv2_m, acd1_m)
    # Compute scale change of all transformations 
    detAff_list = det_acd(Aff_list)
    # Test all hypothesis 
    #====================
    # ---- DEBUG ----
    '''
    INTERACTIVE = True
    TEST = [85, 91, 94, 200, 322, 318, 310]
    BAD_SCALE = [89, 319]
    hypothesis_iter = iter(TEST)
    '''
    #====================
    for mx in reversed(xrange(len(x1_m))):
        # --- Get the mth hypothesis ---
        A11 = Aff_list[0,mx]
        A21 = Aff_list[1,mx]
        A22 = Aff_list[2,mx]
        Adet = detAff_list[mx]
        x1_hypo = x1_m[mx]
        x2_hypo = x2_m[mx]
        y1_hypo = y1_m[mx]
        y2_hypo = y2_m[mx]
        # --- Transform from kpts1 to kpts2 ---
        x1_mt   = x2_hypo + A11*(x1_m - x1_hypo)
        y1_mt   = y2_hypo + A21*(x1_m - x1_hypo) + A22*(y1_m - y1_m[mx])
        det1_mt = det1_m / Adet
        # --- Find (Squared) Error ---
        xy_err    = xy_error_acd(x1_m, y1_mt, x2_m, y2_m) 
        scale_err = det1_mt / det2_m
        # --- Determine Inliers ---
        xy_inliers = xy_err < xy_thresh_sqrd 
        scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                       scale_err < scale_thresh_high)
        hypo_inliers, = np.where(np.logical_and(xy_inliers, scale_inliers))
        #====================
        # ---- DEBUG ----
        '''
        print(' --- Get the mth=%d hypothesis' % (mx))
        print('num_xy_inliers= = %r' % (xy_inliers.sum()))
        print('num_scale_inliers = %r' % (scale_inliers.sum()))
        print('num_total_inliers = %r' % (len(hypo_inliers)))
        '''
        #====================
        # --- Update Best Inliers ---
        if len(hypo_inliers) > len(best_inliers):
            print('# new inliers = %d ' % (len(hypo_inliers)))
            best_inliers = hypo_inliers
            best_Aff     = Aff_list[:, mx]
        #====================
        # ---- DEBUG ----
        '''
        if INTERACTIVE:
            # Transform my compact arrays into a readable form
            def acd2_mat(acd):
                a, c, d = acd
                return np.array([(a, 0), (c, d)])
            A1 = acd2_mat(acd1_m[:, mx])
            A2 = acd2_mat(acd2_m[:, mx])
            A_mine = acd2_mat(Aff_list[:, mx])
            A_numpy = scipy.linalg.inv(A2).dot(A1)
            A1_t = A_mine.dot(A1)
            def matstr(A):
                accur = 4
                F = '%.'+str(accur)+'f'
                matstr = ('\n ['+F+', '+F+' ;\n  '+F+', '+F+']') % tuple(A.flatten().tolist())
                det_ = np.sqrt(1/np.linalg.det(A))
                detstr = '\n 1/sqrt(det)='+str(det_)
                return matstr + detstr
            # Print out current shape hypothesis
            print('Correct way to calculate xy transform: ')
            print('A1 tranforms the ellipse in img1 to a unit circle')
            print('inv(A2) tranforms a unit circle to an ellipse in img2')
            print('A1 = '+matstr(A1))
            print('A2 = '+matstr(A2))
            print('---')
            print('A = inv(A2).dot(A)')
            print('---')
            print('A  = '+matstr(A_mine)+' #mycalc')
            print('A  = '+matstr(A_numpy)+' #npcalc')
            print('---')
            print('A.dot(A1) = '+matstr(A1_t)+' #mycalc')
            print('---')

            det1_ = np.linalg.det(A1)
            det2_ = np.linalg.det(A2)
            detA_ = np.linalg.det(A_mine)

            det1_si = np.sqrt(1/det1_)
            det2_si = np.sqrt(1/det2_)
            detA_si = np.sqrt(1/detA_)
            print('---')
            print('Correct way to calculate scale error: ')
            print('1/sqrt(det)\'s=')
            print('det1_si= %.2f' % det1_si)
            print('det2_si= %.2f' % det2_si)
            print('detA_si= %.2f' % detA_si)
            print('(detA_si:%.1f) * (det2_si:%.1f) == (det1_si:%.1f)'  % (detA_si, det2_si, det1_si))
            print('(detA_si:%.1f) * (det2_si:%.1f) = %.2f' % (detA_si, det2_si, det2_si * detA_si))
            print('--- Therefore')
            print('squaring both sides')
            print('detA_si**2 * det2_si**2 = det1_si**2')
            print('%.1f * %.1f = %.1f' % (detA_si**2, det2_si**2, det1_si**2))
            print('%.1f = %.1f' % (detA_si**2 * det2_si**2, det1_si**2))
            print('...True')
            print('Inverting')
            print('(detA_:%r) * (det2_:%r) = (det1_:%r)'  % (detA_, det2_, det1_)) 
            print('(detA_*det2_:%r) = (det1_:%r)'  % (detA_ * det2_, det1_)) 
            print('(det1_/detA_:%r) = (det2_:%r)'  % (det1_ / detA_, det2_)) 
            print('det1_/detA_ = det2_')
            print('...True')

            df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm[[mx]], fs=None,
                            all_kpts=False, draw_lines=False, doclf=True,
                            title='Testing mth=%d match' % mx)
            df2.update()
            ans = raw_input('press enter to continue (q to quit, b to break)')
            if ans == 'q':
                INTERACTIVE = False
            if ans == 'b':
                break
        '''
        #====================
    return best_inliers, best_Aff
    
    

    
def __affine_inliers(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
    with util.Timer(msg='both') as t:
        best_inliers = []
        best_Aff = None
        # Get keypoint scales (determinant)
        det1_m = det_acd(acd1_m)
        det2_m = det_acd(acd2_m)
        # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
        inv2_m = inv_acd(acd2_m, det2_m)
        # The transform from kp1 to kp2 is given as:
        # A = inv(A2).dot(A1)
        Aff_list = dot_acd(inv2_m, acd1_m)
        # Compute scale change of all transformations 
        detAff_list = det_acd(Aff_list)
        # Test all hypothesis 
        for mx in reversed(xrange(len(x1_m))):
            # --- Get the mth hypothesis ---
            A11 = Aff_list[0,mx]
            A21 = Aff_list[1,mx]
            A22 = Aff_list[2,mx]
            Adet = detAff_list[mx]
            x1_hypo = x1_m[mx]
            x2_hypo = x2_m[mx]
            y1_hypo = y1_m[mx]
            y2_hypo = y2_m[mx]
            # --- Transform from kpts1 to kpts2 ---
            x1_mt   = x2_hypo + A11*(x1_m - x1_hypo)
            y1_mt   = y2_hypo + A21*(x1_m - x1_hypo) + A22*(y1_m - y1_m[mx])
            det1_mt = det1_m / Adet
            # --- Find (Squared) Error ---
            xy_err    = xy_error_acd(x1_mt, y1_mt, x2_m, y2_m) 
            scale_err = det1_mt / det2_m
            # --- Determine Inliers ---
            xy_inliers = xy_err < xy_thresh_sqrd 
            scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                        scale_err < scale_thresh_high)
            hypo_inliers, = np.where(np.logical_and(xy_inliers, scale_inliers))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                print('# new inliers = %d ' % (len(hypo_inliers)))
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]
    return best_inliers, best_Aff

#mx2_num_inliers = mx2_scale_inliers.sum(axis=0)
#top_mx = mx2_num_inliers.argsort()[::-1]
#print mx2_num_inliers[mx2_num_inliers.argsort()[-1]
    
#print(' --- mx=%d ---' % mx)
#print('# scale inliers = %d ' % (mx2_scale_inliers[mx].sum()))
#print('# xy inliers = %d ' % (xy_inliers.sum()))
#print('# total inliers = %d ' % (len(hypo_inliers)))


# .017
# .014

def __affine_inliers(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Precompute transformations for each correspondence
    inv2_m = inv_acd(acd2_m, det2_m)
    Aff_list = dot_acd(inv2_m, acd1_m) # A = inv(A2).dot(A1) 
    detAff_list = det_acd(Aff_list)    # detA = det1 / det2

    # Step 1: Compute scale inliers
    detAff_list.shape = (1, detAff_list.size)
    det2_m.shape = (det2_m.size, 1)
    det1_m.shape = (det1_m.size, 1)
    scale_err_mat = (det2_m/det1_m).dot(detAff_list)
    mx2_scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                        scale_err < scale_thresh_high)
    # Step 2: Compute spatial inliers -- this can be cascaded based on scale

    # Get all x_err
    # Get all y_err
    AffT = Aff_list.T
    mx2_xyerr = np.vstack(
        [(x1_m - (x2_m[mx] + AffT[mx,0]*(x1_m - x1_m[mx]))) ** 2 +\
            (y1_m - (y2_m[mx] + AffT[mx,1]*(x1_m - x1_m[mx]) +\
                                AffT[mx,2]*(y1_m - y1_m[mx]))) ** 2
                                for mx in xrange(len(x1_m))])

    mx2_xy_inliers = mx2_xyerr < xy_thresh_sqrd
    mx2_inliers = np.logical_and(mx2_scale_inliers, mx2_xy_inliers)
    mx2_num_inliers = mx2_inliers.sum(axis=0)
    best_mx = mx2_num_inliers.argsort()[-1]
    best_Aff = AffT[best_mx]
    best_inliers = mx2_inliers[best_mx]



    test3 = '''
        best_inliers = []
        best_Aff = None
        Aff_listT = Aff_list.T
        for x1_hypo, x2_hypo, y1_hypo, y2_hypo, Aff in izip(x1_m, x2_m, y1_m, y2_m, Aff_listT):
            x1_mt   = x1_hypo + Aff[0]*(x1_m - x1_hypo)
            y1_mt   = y2_hypo + Aff[1]*(x1_m - x1_hypo) + Aff[2]*(y1_m - y1_hypo)
            # --- Find (Squared) Error ---
            xy_err    = xy_error_acd(x1_mt, y1_mt, x2_m, y2_m) 
            # --- Determine Inliers ---
        for mx in xrange(len(x1_m)):
            xy_inliers = xy_err < xy_thresh_sqrd 
            hypo_inliers, = np.where(np.logical_and(xy_inliers, mx2_scale_inliers[mx]))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]
            # --- Transform from kpts1 to kpts2 ---
    '''

    test3 = '''
        best_inliers = []
        best_Aff = None
        Aff_listT = Aff_list.T
        for mx in xrange(len(x1_m)):
            Aff = Aff_listT
            # --- Transform from kpts1 to kpts2 ---
            x1_mt   = x2_m[mx] + Aff[mx, 0]*(x1_m - x1_m[mx])
            y1_mt   = y2_m[mx] + Aff[mx, 1]*(x1_m - x1_m[mx]) + Aff[mx, 2]*(y1_m - y1_m[mx])
            # --- Find (Squared) Error ---
            xy_err    = (x1_mt - x2_m)**2 + (y1_mt - y2_m)**2
            # --- Determine Inliers ---
    '''

    test3 = '''
        best_inliers = []
        best_Aff = None
    '''
    print timeit.timeit(test3, setup=setup_, number=200)

        for mx in xrange(len(x1_m)):
            xy_inliers = xy_err < xy_thresh_sqrd 
            hypo_inliers, = np.where(np.logical_and(xy_inliers, mx2_scale_inliers[mx]))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]

        
    
    local_dict = locals().copy()
    exclude_list=['_*', 'In', 'Out', 'rchip1', 'rchip2', 'nan', 'inf', 'Inf']
    util.rrr()
    setup = util.execstr_timeitsetup(local_dict, exclude_list)

    import textwrap
    import timeit
    setup_ = textwrap.dedent('''
    from numpy import array, float32, float64, int32, uint32, int64, inf
    def xy_error_acd(x1, y1, x2, y2):
        'Aligned points spatial error'
        return (x1 - x2)**2 + (x1 - y1)**2
    from itertools import izip
    ''') + setup

    print timeit.timeit(test1, setup=setup_, number=500)
    print timeit.timeit(test2, setup=setup_, number=500)
    print timeit.timeit(test3, setup=setup_, number=500)


    with util.Timer(msg='stack'):
    test1 = '''
        best_inliers = []
        best_Aff = None
        for mx in xrange(len(x1_m)):
            # --- Get the mth hypothesis ---
            A11 = Aff_list[0,mx]
            A21 = Aff_list[1,mx]
            A22 = Aff_list[2,mx]
            x1_hypo = x1_m[mx]
            x2_hypo = x2_m[mx]
            y1_hypo = y1_m[mx]
            y2_hypo = y2_m[mx]
            # --- Transform from kpts1 to kpts2 ---
            x1_mt   = x2_hypo + A11*(x1_m - x1_hypo)
            y1_mt   = y2_hypo + A21*(x1_m - x1_hypo) + A22*(y1_m - y1_m[mx])
            # --- Find (Squared) Error ---
            xy_err    = xy_error_acd(x1_mt, y1_mt, x2_m, y2_m) 
            # --- Determine Inliers ---
            xy_inliers = xy_err < xy_thresh_sqrd 
            hypo_inliers, = np.where(np.logical_and(xy_inliers, mx2_scale_inliers[mx]))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]
    '''
                
    with util.Timer(msg='randacc'):
    test2 = '''
        best_inliers = []
        best_Aff = None
        Aff = Aff_list.T
        for mx in xrange(len(x1_m)):
            # --- Get the mth hypothesis ---
            A11 = Aff[mx, 0]
            A21 = Aff[mx, 1]
            A22 = Aff[mx, 2]
            # --- Transform from kpts1 to kpts2 ---
            x1_mt   = x2_m[mx] + A11*(x1_m - x1_m[mx])
            y1_mt   = y2_m[mx] + A21*(x1_m - x1_m[mx]) + A22*(y1_m - y1_m[mx])
            # --- Find (Squared) Error ---
            xy_err    = xy_error_acd(x1_mt, y1_mt, x2_m, y2_m) 
            # --- Determine Inliers ---
            xy_inliers = xy_err < xy_thresh_sqrd 
            hypo_inliers, = np.where(np.logical_and(xy_inliers, mx2_scale_inliers[mx]))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]
    '''

    test3 = '''
        best_inliers = []
        best_Aff = None
        Aff = Aff_list.T
        for mx in xrange(len(x1_m)):
            # --- Transform from kpts1 to kpts2 ---
            x1_mt   = x2_m[mx] + Aff[mx, 0]*(x1_m - x1_m[mx])
            y1_mt   = y2_m[mx] + Aff[mx, 1]*(x1_m - x1_m[mx]) + Aff[mx, 2]*(y1_m - y1_m[mx])
            # --- Find (Squared) Error ---
            xy_err    = xy_error_acd(x1_mt, y1_mt, x2_m, y2_m) 
            # --- Determine Inliers ---
        for mx in xrange(len(x1_m)):
            xy_inliers = xy_err < xy_thresh_sqrd 
            hypo_inliers, = np.where(np.logical_and(xy_inliers, mx2_scale_inliers[mx]))
            # --- Update Best Inliers ---
            if len(hypo_inliers) > len(best_inliers):
                best_inliers = hypo_inliers
                best_Aff     = Aff_list[:, mx]
    '''

    return best_inliers, best_Aff




# The old one is faster. Bullshit
def __affine_inliers_needs_work(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Precompute transformations for each correspondence
    inv2_m = inv_acd(acd2_m, det2_m)
    Aff_list = dot_acd(inv2_m, acd1_m) # A = inv(A2).dot(A1) 
    detAff_list = det_acd(Aff_list)    # detA = det1 / det2
    #
    detAff_list.shape = (1, detAff_list.size)
    det2_m.shape = (det2_m.size, 1)
    det1_m.shape = (det1_m.size, 1)
    AffT = Aff_list.T
    # Step 1: Compute scale inliers
    scale_err_mat = (det2_m/det1_m).dot(detAff_list)
    # Step 2: Compute spatial inliers -- this can be cascaded based on scale
    # breaking a core tenant of python for some extra speed
    mx2_xyerr = np.array([
        (x2_m - (x2_m[mx] + AffT[mx,0]*(x1_m - x1_m[mx]))) ** 2 +\
        (y2_m - (y2_m[mx] + AffT[mx,1]*(x1_m - x1_m[mx])        +\
                            AffT[mx,2]*(y1_m - y1_m[mx]))) ** 2
        for mx in xrange(len(x1_m))])

    mx2_scale_inliers = np.logical_and(scale_err_mat > scale_thresh_low,
                                       scale_err_mat < scale_thresh_high)
    mx2_xy_inliers = mx2_xyerr < xy_thresh_sqrd
    # axis=1 might cause issues. bytes-order is different on hyrule
    mx2_inliers = np.logical_and(mx2_scale_inliers, mx2_xy_inliers)
    mx2_num_inliers = mx2_inliers.sum(axis=1)
    #
    best_mx = mx2_num_inliers.argsort()[-1]
    best_Aff = AffT[best_mx]
    best_inliers = np.where(mx2_inliers[best_mx])[0]
    return best_inliers, best_Aff

cmd = '__affine_inliers_needs_work(x1_m, y1_m, acd1_m, x2_m, y2_m, acd2_m, xy_thresh_sqrd, scale_thresh_high, scale_thresh_low)'


    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers1], kpts2_m.T[aff_inliers1], title=title, fignum=2, vert=False)


def __affine_inliers_cascade(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
#with util.Timer('scale cascade'):
    best_inliers = []
    best_Aff = None
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    inv2_m = inv_acd(acd2_m, det2_m)
    # The transform from kp1 to kp2 is given as:
    # A = inv(A2).dot(A1)
    Aff_list = dot_acd(inv2_m, acd1_m)
    # Compute scale change of all transformations 
    detAff_list = det_acd(Aff_list)
    # Test all hypothesis 
    for mx in xrange(len(x1_m)):
        # --- Get the mth hypothesis ---
        A11 = Aff_list[0,mx]
        A21 = Aff_list[1,mx]
        A22 = Aff_list[2,mx]
        Adet = detAff_list[mx]
        x1_hypo = x1_m[mx]
        x2_hypo = x2_m[mx]
        y1_hypo = y1_m[mx]
        y2_hypo = y2_m[mx]
        # --- Determine Scale Inliers ---
        scale_err = Adet * det2_m / det1_m
        scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                       scale_err < scale_thresh_high)
        # Dont test non-scale-inliers for spatial consistency
        x1_ms = x1_m[scale_inliers]
        y1_ms = y1_m[scale_inliers]
        x2_ms = x2_m[scale_inliers]
        y2_ms = y2_m[scale_inliers]
        # --- Transform from kpts1 to kpts2 ---
        x1_mst = x2_hypo + A11*(x1_ms - x1_hypo)
        y1_mst = y2_hypo + A21*(x1_ms - x1_hypo) +\
                           A22*(y1_ms - y1_hypo)
        # --- Find (Squared) XY-Error ---
        xy_err     = (x1_mst - x2_ms)**2 + (y1_mst - y2_ms)**2 
        xy_inliers = xy_err < xy_thresh_sqrd
        # --- Inliers in xy and scale ---
        hypo_inliers = np.where(scale_inliers)[0][xy_inliers]
        # --- Update Best Inliers ---
        if len(hypo_inliers) > len(best_inliers):
            best_inliers = hypo_inliers
            best_Aff     = Aff_list[:, mx]
    return best_inliers, best_Aff

    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers2], kpts2_m.T[aff_inliers2], title=title, fignum=3, vert=False)


def affine_inliers(kpts1, kpts2, fm, xy_thresh, scale_thresh):
    scale_thresh_low  = scale_thresh
    scale_thresh_high = 1.0 / scale_thresh_low
    # Get matching keypoints (x, y, ellipse_acd)
    x1_m, y1_m, acd1_m = split_kpts(kpts1[fm[:, 0]].T)
    x2_m, y2_m, acd2_m = split_kpts(kpts2[fm[:, 1]].T)

    # TODO: Pass in the diag length
    x2_extent = x2_m.max() - x2_m.min()
    y2_extent = y2_m.max() - y2_m.min()
    img2_diaglen_sqrd = x2_extent**2 + y2_extent**2
    xy_thresh_sqrd = img2_diaglen_sqrd * xy_thresh

    inliers, Aff = __affine_inliers(x1_m, y1_m, acd1_m,
                                    x2_m, y2_m, acd2_m, xy_thresh_sqrd, 
                                    scale_thresh_high, scale_thresh_low)
    return inliers, Aff



def test_realdata2():
    from util import printWARN, printINFO
    import warnings
    import numpy.linalg as linalg
    import numpy as np
    import scipy.sparse as sparse
    import scipy.sparse.linalg as sparse_linalg
    import load_data2
    import params
    import draw_func2 as df2
    import util
    import spatial_verification
    #params.reload_module()
    #load_data2.reload_module()
    #df2.reload_module()

    db_dir = load_data2.MOTHERS
    hs = load_data2.HotSpotter(db_dir)
    assign_matches = hs.matcher.assign_matches
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    # Get chips
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    # Get feature matches 
    kpts1_m = kpts1[fm[:, 0], :].T
    kpts2_m = kpts2[fm[:, 1], :].T
    
    title='(qx%r v cx%r)\n #match=%r' % (qcx, cx, len(fm))
    df2.show_matches2(rchip1, rchip2, kpts1,  kpts2, fm, fs, title=title)

    np.random.seed(6)
    subst = util.random_indexes(len(fm),len(fm))
    kpts1_m = kpts1[fm[subst, 0], :].T
    kpts2_m = kpts2[fm[subst, 1], :].T

    df2.reload_module()
    df2.SHOW_LINES = True
    df2.ELL_LINEWIDTH = 2
    df2.LINE_ALPHA = .5
    df2.ELL_ALPHA  = 1
    df2.reset()
    df2.show_keypoints(rchip1, kpts1_m.T, fignum=0, plotnum=121)
    df2.show_keypoints(rchip2, kpts2_m.T, fignum=0, plotnum=122)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T,  kpts2_m.T, title=title,
                      fignum=1, vert=True)

    spatial_verification.reload_module()
    with util.Timer():
        aff_inliers1 = spatial_verification.aff_inliers_from_ellshape2(kpts1_m, kpts2_m, xy_thresh_sqrd)
    with util.Timer():
        aff_inliers2 = spatial_verification.aff_inliers_from_ellshape(kpts1_m, kpts2_m, xy_thresh_sqrd)

    # Homogonize+Normalize
    xy1_m    = kpts1_m[0:2,:] 
    xy2_m    = kpts2_m[0:2,:]
    (xyz_norm1, T1) = spatial_verification.homogo_normalize_pts(xy1_m[:,aff_inliers1]) 
    (xyz_norm2, T2) = spatial_verification.homogo_normalize_pts(xy2_m[:,aff_inliers1])

    H_prime = spatial_verification.compute_homog(xyz_norm1, xyz_norm2)
    H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize

    Hdet = linalg.det(H)

    # Estimate final inliers
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Precompute the determinant of lower triangular matrix (a*d - b*c); b = 0
    det1_m = acd1_m[0] * acd1_m[2]
    det2_m = acd2_m[0] * acd2_m[2]

    # Matrix Multiply xyacd matrix by H
    # [[A, B, X],      
    #  [C, D, Y],      
    #  [E, F, Z]] 
    # dot 
    # [(a, 0, x),
    #  (c, d, y),
    #  (0, 0, 1)] 
    # = 
    # [(a*A + c*B + 0*E,   0*A + d*B + 0*X,   x*A + y*B + 1*X),
    #  (a*C + c*D + 0*Y,   0*C + d*D + 0*Y,   x*C + y*D + 1*Y),
    #  (a*E + c*F + 0*Z,   0*E + d*F + 0*Z,   x*E + y*F + 1*Z)]
    # =
    # [(a*A + c*B,               d*B,         x*A + y*B + X),
    #  (a*C + c*D,               d*D,         x*C + y*D + Y),
    #  (a*E + c*F,               d*F,         x*E + y*F + Z)]
    # # IF x=0 and y=0
    # =
    # [(a*A + c*B,               d*B,         0*A + 0*B + X),
    #  (a*C + c*D,               d*D,         0*C + 0*D + Y),
    #  (a*E + c*F,               d*F,         0*E + 0*F + Z)]
    # =
    # [(a*A + c*B,               d*B,         X),
    #  (a*C + c*D,               d*D,         Y),
    #  (a*E + c*F,               d*F,         Z)]
    # --- 
    #  A11 = a*A + c*B
    #  A21 = a*C + c*D
    #  A31 = a*E + c*F
    #  A12 = d*B
    #  A22 = d*D
    #  A32 = d*F
    #  A31 = X
    #  A32 = Y
    #  A33 = Z
    #
    # det(A) = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)

    det1_mAt = det1_m * Hdet
    # Check Error in position and scale
    xy_sqrd_err = (x1_mAt - x2_m)**2 + (y1_mAt - y2_m)**2
    scale_sqrd_err = det1_mAt / det2_m
    # Check to see if outliers are within bounds
    xy_inliers = xy_sqrd_err < xy_thresh_sqrd
    s1_inliers = scale_sqrd_err > scale_thresh_low
    s2_inliers = scale_sqrd_err < scale_thresh_high
    _inliers, = np.where(np.logical_and(np.logical_and(xy_inliers, s1_inliers), s2_inliers))

    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd



    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers1], kpts2_m.T[aff_inliers1], title=title, fignum=2, vert=False)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers2], kpts2_m.T[aff_inliers2], title=title, fignum=3, vert=False)
    df2.present(wh=(600,400))


def test():
    weird_A = np.array([(2.7, 4.2, 10),
                        (1.7, 3.1, 20),
                        (2.2, 2.4, 1.1)])

    num_mat = 2000
    frac_in = .5
    num_inl = int(num_mat * frac_in)
    kpts1_m = np.random.rand(5, num_mat)
    kpts2_m = np.random.rand(5, num_mat)

    kpts2_cor = kpts1_m[:, 0:num_inl]

    x_cor = kpts2_cor[0]
    y_cor = kpts2_cor[1]
    z_cor = np.ones(num_inl)
    a_cor = kpts2_cor[2]
    zeros = np.zeros(num_inl)
    c_cor = kpts2_cor[3]
    d_cor = kpts2_cor[4]

    kpts2_mats = np.array([(a_cor, zeros, x_cor),
                           (c_cor, d_cor, y_cor), 
                           (zeros, zeros, z_cor)]).T

    # Do some weird transform on some keypoints
    import scipy.linalg
    for x in xrange(num_inl):
        mat = weird_A.dot(kpts2_mats[x].T)
        mat /= mat[2,2]
        kpts2_m[0,x] = mat[0,2]
        kpts2_m[1,x] = mat[1,2]
        kpts2_m[2,x] = mat[0,0]
        kpts2_m[3,x] = mat[1,0]
        kpts2_m[4,x] = mat[1,2]

    # Find some inliers? 
    xy_thresh_sqrd = 7
    H, inliers = H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)


    #aff_inliers1 = sv2.__affine_inliers(x1_m, y1_m, x2_m, y2_m, 
                                        #acd1_m, acd2_m, xy_thresh_sqrd, scale_thresh)

    H_prime = sv2.compute_homog(x1_mn, y1_mn, x2_mn, y2_mn)
    H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize

    Hdet = linalg.det(H)

    # Estimate final inliers
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Precompute the determinant of lower triangular matrix (a*d - b*c); b = 0
    det1_m = acd1_m[0] * acd1_m[2]
    det2_m = acd2_m[0] * acd2_m[2]

    # Matrix Multiply xyacd matrix by H
    # [[A, B, X],      
    #  [C, D, Y],      
    #  [E, F, Z]] 
    # dot 
    # [(a, 0, x),
    #  (c, d, y),
    #  (0, 0, 1)] 
    # = 
    # [(a*A + c*B + 0*E,   0*A + d*B + 0*X,   x*A + y*B + 1*X),
    #  (a*C + c*D + 0*Y,   0*C + d*D + 0*Y,   x*C + y*D + 1*Y),
    #  (a*E + c*F + 0*Z,   0*E + d*F + 0*Z,   x*E + y*F + 1*Z)]
    # =
    # [(a*A + c*B,               d*B,         x*A + y*B + X),
    #  (a*C + c*D,               d*D,         x*C + y*D + Y),
    #  (a*E + c*F,               d*F,         x*E + y*F + Z)]
    # # IF x=0 and y=0
    # =
    # [(a*A + c*B,               d*B,         0*A + 0*B + X),
    #  (a*C + c*D,               d*D,         0*C + 0*D + Y),
    #  (a*E + c*F,               d*F,         0*E + 0*F + Z)]
    # =
    # [(a*A + c*B,               d*B,         X),
    #  (a*C + c*D,               d*D,         Y),
    #  (a*E + c*F,               d*F,         Z)]
    # --- 
    #  A11 = a*A + c*B
    #  A21 = a*C + c*D
    #  A31 = a*E + c*F
    #  A12 = d*B
    #  A22 = d*D
    #  A32 = d*F
    #  A31 = X
    #  A32 = Y
    #  A33 = Z
    #
    # det(A) = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)

    det1_mAt = det1_m * Hdet
    # Check Error in position and scale
    xy_sqrd_err = (x1_mAt - x2_m)**2 + (y1_mAt - y2_m)**2
    scale_sqrd_err = det1_mAt / det2_m
    # Check to see if outliers are within bounds
    xy_inliers = xy_sqrd_err < xy_thresh_sqrd
    s1_inliers = scale_sqrd_err > scale_thresh_low
    s2_inliers = scale_sqrd_err < scale_thresh_high
    _inliers, = np.where(np.logical_and(np.logical_and(xy_inliers, s1_inliers), s2_inliers))

    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd



    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers1], kpts2_m.T[aff_inliers1], title=title, fignum=2, vert=False)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers2], kpts2_m.T[aff_inliers2], title=title, fignum=3, vert=False)
    df2.present(wh=(600,400))

# This new function is much faster .035 vs .007
    # EXPLOITS LOWER TRIANGULAR MATRIXES
    # Precompute the determinant of matrix 2 (a*d - b*c), but b = 0
    # Need the inverse of acd2_m:  1/det * [(d, -b), (-c, a)]
    # Precompute lower triangular affine tranforms inv2_m (dot) acd1_m
    # [(a2*a1), (c2*a1+d2*c1), (d2*d1)]
    
    # IN HOMOGRAPHY INLIERS
    '''
    y_data = []
    x_data = []
    for xy_thresh in np.linspace(.001, .005, 50):
        xy_thresh_sqrd = img2_diaglen_sqrd * xy_thresh
        aff_inliers = __affine_inliers(x1_m, y1_m, acd1_m, 
                                    x2_m, y2_m, acd2_m,
                                    xy_thresh_sqrd, 
                                    scale_thresh_low,
                                    scale_thresh_high)
        y_data.append(len(aff_inliers))
        x_data.append(xy_thresh)

    plt.plot(x_data, y_data)


    df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm=fm[aff_inliers,:],
                      fignum=3, vert=True)    
    '''

    '''
    df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm=fm[hom_inliers,:],
                      fignum=3, vert=True)    
    '''
