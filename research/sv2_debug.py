    
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
    with helpers.Timer(msg='both') as t:
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
    exclude_list=['_*', 'In', 'Out', 'rchip1', 'rchip2']
    helpers.rrr()
    setup = helpers.execstr_timeitsetup(local_dict, exclude_list)

    setup_ = textwrap.dedent('''
    from numpy import array, float32, float64, int32, uint32, int64
    def xy_error_acd(x1, y1, x2, y2):
        'Aligned points spatial error'
        return (x1 - x2)**2 + (x1 - y1)**2
    from itertools import izip
    ''') + setup

    print timeit.timeit(test1, setup=setup_, number=500)
    print timeit.timeit(test2, setup=setup_, number=500)
    print timeit.timeit(test3, setup=setup_, number=500)


    with helpers.Timer(msg='stack'):
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
                
    with helpers.Timer(msg='randacc'):
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
