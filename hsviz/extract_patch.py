#exec(open('__init__.py').read())
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[extract]')
# Science
import cv2
import numpy as np
from numpy import sqrt
# Hotspotter
from . import draw_func2 as df2


def rrr():
    import imp
    import sys
    print('[extract] Reloading: ' + __name__)
    imp.reload(sys.modules[__name__])


def svd(M):
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U, S, V


def draw_warped_keypoint_patch(rchip, kp, **kwargs):
    return draw_keypoint_patch(rchip, kp, warped=True, **kwargs)


def draw_keypoint_patch(rchip, kp, sift=None, warped=False, **kwargs):
    #print('--------------------')
    #print('[extract] Draw Patch')
    if warped:
        wpatch, wkp = get_warped_patch(rchip, kp)
        patch = wpatch
        subkp = wkp
    else:
        patch, subkp = get_patch(rchip, kp)
    #print('[extract] kp    = '+str(kp))
    #print('[extract] subkp = '+str(subkp))
    #print('[extract] patch.shape = %r' % (patch.shape,))
    color = (0, 0, 1)
    fig, ax = df2.imshow(patch, **kwargs)
    df2.draw_kpts2([subkp], ell_color=color, pts=True)
    if not sift is None:
        df2.draw_sift(sift, [subkp])
    return ax
    #df2.draw_border(df2.gca(), color, 1)


def get_aff_to_unit_circle(a, c, d):
    invA = np.array([[a, 0, 0],
                     [c, d, 0],
                     [0, 0, 1]])
    # kp is given in invA format. Convert to A
    A = np.linalg.inv(invA)
    return A


def get_translation(x, y):
    T = np.array([[1, 0,  x],
                  [0, 1,  y],
                  [0, 0,  1]])
    return T


def get_scale(ss):
    S = np.array([[ss, 0, 0],
                  [0, ss, 0],
                  [0,  0, 1]])
    return S


def get_warped_patch(rchip, kp):
    'Returns warped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy = kp2_sf(kp)
    s = 41  # sf
    ss = sqrt(s) * 3
    (h, w) = rchip.shape[0:2]
    # Translate to origin(0,0) = (x,y)
    T = get_translation(-x, -y)
    A = get_aff_to_unit_circle(a, c, d)
    S = get_scale(ss)
    X = get_translation(s / 2, s / 2)
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    M = X.dot(S).dot(A).dot(T)
    cv2_flags = cv2.INTER_LANCZOS4
    cv2_borderMode = cv2.BORDER_CONSTANT
    cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}
    warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **cv2_warp_kwargs)
    #warped_patch = cv2.warpPerspective(rchip, M, dsize, **__cv2_warp_kwargs())
    wkp = np.array([(s / 2, s / 2, ss, 0., ss)])
    return warped_patch, wkp


def in_depth_ellipse2x2(rchip, kp):
    #-----------------------
    # SETUP
    #-----------------------
    from hotspotter import draw_func2 as df2
    np.set_printoptions(precision=8)
    tau = 2 * np.pi
    df2.reset()
    df2.figure(9003, docla=True, doclf=True)
    ax = df2.gca()
    ax.invert_yaxis()

    def _plotpts(data, px, color=df2.BLUE, label=''):
        #df2.figure(9003, docla=True, pnum=(1, 1, px))
        df2.plot2(data.T[0], data.T[1], '.', '', color=color, label=label)
        df2.update()

    def _plotarrow(x, y, dx, dy, color=df2.BLUE, label=''):
        ax = df2.gca()
        arrowargs = dict(head_width=.5, length_includes_head=True, label=label)
        arrow = df2.FancyArrow(x, y, dx, dy, **arrowargs)
        arrow.set_edgecolor(color)
        arrow.set_facecolor(color)
        ax.add_patch(arrow)
        df2.update()

    def _2x2_eig(M2x2):
        (evals, evecs) = np.linalg.eig(M2x2)
        l1, l2 = evals
        v1, v2 = evecs
        return l1, l2, v1, v2

    #-----------------------
    # INPUT
    #-----------------------
    # We will call perdoch's invA = invV
    print('--------------------------------')
    print('Let V = Perdoch.A')
    print('Let Z = Perdoch.E')
    print('--------------------------------')
    print('Input from Perdoch\'s detector: ')

    # We are given the keypoint in invA format
    (x, y, ia11, ia21, ia22), ia12 = kp, 0
    invV = np.array([[ia11, ia12],
                     [ia21, ia22]])
    V = np.linalg.inv(invV)
    # <HACK>
    #invV = V / np.linalg.det(V)
    #V = np.linalg.inv(V)
    # </HACK>
    Z = (V.T).dot(V)

    print('invV is a transform from points on a unit-circle to the ellipse')
    helpers.horiz_print('invV = ', invV)
    print('--------------------------------')
    print('V is a transformation from points on the ellipse to a unit circle')
    helpers.horiz_print('V = ', V)
    print('--------------------------------')
    print('Points on a matrix satisfy (x).T.dot(Z).dot(x) = 1')
    print('where Z = (V.T).dot(V)')
    helpers.horiz_print('Z = ', Z)

    # Define points on a unit circle
    theta_list = np.linspace(0, tau, 50)
    cicrle_pts = np.array([(np.cos(t), np.sin(t)) for t in theta_list])

    # Transform those points to the ellipse using invV
    ellipse_pts1 = invV.dot(cicrle_pts.T).T

    # Transform those points to the ellipse using V
    ellipse_pts2 = V.dot(cicrle_pts.T).T

    #Lets check our assertion: (x_).T.dot(Z).dot(x_) = 1
    checks1 = [x_.T.dot(Z).dot(x_) for x_ in ellipse_pts1]
    checks2 = [x_.T.dot(Z).dot(x_) for x_ in ellipse_pts2]
    assert all([abs(1 - check) < 1E-11 for check in checks1])
    #assert all([abs(1 - check) < 1E-11 for check in checks2])
    print('... all of our plotted points satisfy this')

    #=======================
    # THE CONIC SECTION
    #=======================
    # All of this was from the Perdoch paper, now lets move into conic sections
    # We will use the notation from wikipedia
    # http://en.wikipedia.org/wiki/Conic_section
    # http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

    #-----------------------
    # MATRIX REPRESENTATION
    #-----------------------
    # The matrix representation of a conic is:
    (A,  B2, B2_, C) = Z.flatten()
    (D, E, F) = (0, 0, 1)
    B = B2 * 2
    assert B2 == B2_, 'matrix should by symmetric'
    print('--------------------------------')
    print('Now, using wikipedia\' matrix representation of a conic.')
    con = np.array((('    A', 'B / 2', 'D / 2'),
                    ('B / 2', '    C', 'E / 2'),
                    ('D / 2', 'E / 2', '    F')))
    helpers.horiz_print('A matrix A_Q = ', con)

    # A_Q is our conic section (aka ellipse matrix)
    A_Q = np.array(((    A, B / 2, D / 2),
                    (B / 2,     C, E / 2),
                    (D / 2, E / 2,     F)))

    helpers.horiz_print('A_Q = ', A_Q)

    #-----------------------
    # DEGENERATE CONICS
    #-----------------------
    print('----------------------------------')
    print('As long as det(A_Q) != it is not degenerate.')
    print('If the conic is not degenerate, we can use the 2x2 minor: A_33')
    print('det(A_Q) = %s' % str(np.linalg.det(A_Q)))
    assert np.linalg.det(A_Q) != 0, 'degenerate conic'
    A_33 = np.array(((    A, B / 2),
                     (B / 2,     C)))
    helpers.horiz_print('A_33 = ', A_33)

    #-----------------------
    # CONIC CLASSIFICATION
    #-----------------------
    print('----------------------------------')
    print('The determinant of the minor classifies the type of conic it is')
    print('(det == 0): parabola, (det < 0): hyperbola, (det > 0): ellipse')
    print('det(A_33) = %s' % str(np.linalg.det(A_33)))
    assert np.linalg.det(A_33) > 0, 'conic is not an ellipse'
    print('... this is indeed an ellipse')

    #-----------------------
    # CONIC CENTER
    #-----------------------
    print('----------------------------------')
    print('the centers of the ellipse are obtained by: ')
    print('x_center = (B * E - (2 * C * D)) / (4 * A * C - B ** 2)')
    print('y_center = (D * B - (2 * A * E)) / (4 * A * C - B ** 2)')
    # Centers are obtained by solving for where the gradient of the quadratic
    # becomes 0. Without going through the derivation the calculation is...
    # These should be 0, 0 if we are at the origin, or our original x, y
    # coordinate specified by the keypoints. I'm doing the calculation just for
    # shits and giggles
    x_center = (B * E - (2 * C * D)) / (4 * A * C - B ** 2)
    y_center = (D * B - (2 * A * E)) / (4 * A * C - B ** 2)
    helpers.horiz_print('x_center = ', x_center)
    helpers.horiz_print('y_center = ', y_center)

    #-----------------------
    # MAJOR AND MINOR AXES
    #-----------------------
    # Now we are going to determine the major and minor axis
    # of this beast. It just the center augmented by the eigenvecs
    print('----------------------------------')

    # The angle between the major axis and our x axis is:
    l1, l2, v1, v2 = _2x2_eig(A_33)
    x_axis = np.array([1, 0])
    theta = np.arccos(x_axis.dot(v1))

    # The eccentricity is determined by:
    nu = 1
    numer  = 2 * np.sqrt((A - C) ** 2 + B ** 2)
    denom  = nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
    eccentricity = np.sqrt(numer / denom)

    from scipy.special import ellipeinc
    #-----------------------
    # DRAWING
    #-----------------------
    # Lets start off by drawing the ellipse that we are goign to work with
    # Create unit circle sample

    # Draw the keypoint using the tried and true df2
    # Other things should subsiquently align
    df2.draw_kpts2(np.array([(0, 0, ia11, ia21, ia22)]), ell_linewidth=4,
                   ell_color=df2.DEEP_PINK, ell_alpha=1, arrow=True, rect=True)

    # Plot ellipse points
    _plotpts(ellipse_pts1, 0, df2.YELLOW, label='invV.dot(cicrle_pts.T).T')

    # Plot ellipse axis
    # !HELP! I DO NOT KNOW WHY I HAVE TO DIVIDE, SQUARE ROOT, AND NEGATE!!!
    l1, l2, v1, v2 = _2x2_eig(A_33)
    dx1, dy1 = (v1 / np.sqrt(l1))
    dx2, dy2 = (v2 / np.sqrt(l2))
    _plotarrow(0, 0, dx1, -dy1, color=df2.ORANGE, label='ellipse axis')
    _plotarrow(0, 0, dx2, -dy2, color=df2.ORANGE)

    # Plot ellipse orientation
    orient_axis = invV.dot(np.eye(2))
    dx1, dx2, dy1, dy2 = orient_axis.flatten()
    _plotarrow(0, 0, dx1, dy1, color=df2.BLUE, label='ellipse rotation')
    _plotarrow(0, 0, dx2, dy2, color=df2.BLUE)

    df2.legend()
    df2.dark_background()
    df2.gca().invert_yaxis()
    return locals()
    # Algebraic form of connic
    #assert (a * (x ** 2)) + (b * (x * y)) + (c * (y ** 2)) + (d * x) + (e * y) + (f) == 0


def get_kp_border(rchip, kp):
    np.set_printoptions(precision=8)

    df2.reset()
    df2.figure(9003, docla=True, doclf=True)

    def _plotpts(data, px, color=df2.BLUE, label=''):
        #df2.figure(9003, docla=True, pnum=(1, 1, px))
        df2.plot2(data.T[0], data.T[1], '-', '', color=color, label=label)
        df2.update()

    def _plotarrow(x, y, dx, dy, color=df2.BLUE, label=''):
        ax = df2.gca()
        arrowargs = dict(head_width=.5, length_includes_head=True, label='')
        arrow = df2.FancyArrow(x, y, dx, dy, **arrowargs)
        arrow.set_edgecolor(color)
        arrow.set_facecolor(color)
        ax.add_patch(arrow)
        df2.update()

    def _2x2_eig(M2x2):
        (evals, evecs) = np.linalg.eig(M2x2)
        l1, l2 = evals
        v1, v2 = evecs
        return l1, l2, v1, v2

    #-----------------------
    # INPUT
    #-----------------------
    # We are given the keypoint in invA format
    (x, y, ia11, ia21, ia22), ia12 = kp, 0

    # invA2x2 is a transformation from points on a unit circle to the ellipse
    invA2x2 = np.array([[ia11, ia12],
                        [ia21, ia22]])

    #-----------------------
    # DRAWING
    #-----------------------
    # Lets start off by drawing the ellipse that we are goign to work with
    # Create unit circle sample
    tau = 2 * np.pi
    theta_list = np.linspace(0, tau, 1000)
    cicrle_pts = np.array([(np.cos(t), np.sin(t)) for t in theta_list])
    ellipse_pts = invA2x2.dot(cicrle_pts.T).T
    _plotpts(ellipse_pts, 0, df2.BLACK, label='invA2x2.dot(unit_circle)')
    l1, l2, v1, v2 = _2x2_eig(invA2x2)
    dx1, dy1 = (v1 * l1)
    dx2, dy2 = (v2 * l2)
    _plotarrow(0, 0, dx1, dy1, color=df2.ORANGE, label='invA2x2 e1')
    _plotarrow(0, 0, dx2, dy2, color=df2.RED, label='invA2x2 e2')

    #-----------------------
    # REPRESENTATION
    #-----------------------
    # A2x2 is a transformation from points on the ellipse to a unit circle
    A2x2 = np.linalg.inv(invA2x2)

    # Points on a matrix satisfy (x).T.dot(E2x2).dot(x) = 1
    E2x2 = A2x2.T.dot(A2x2)

    #Lets check our assertion: (x).T.dot(E2x2).dot(x) = 1
    checks = [pt.T.dot(E2x2).dot(pt) for pt in ellipse_pts]
    assert all([abs(1 - check) < 1E-11 for check in checks])

    #-----------------------
    # CONIC SECTIONS
    #-----------------------
    # All of this was from the Perdoch paper, now lets move into conic sections
    # We will use the notation from wikipedia
    # http://en.wikipedia.org/wiki/Conic_section
    # http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

    # The matrix representation of a conic is:
    ((A,  B, B_, C), (D, E, F)) = (E2x2.flatten(), (0, 0, 1))
    assert B == B_, 'matrix should by symmetric'

    # A_Q is our conic section (aka ellipse matrix)
    A_Q = np.array(((    A, B / 2, D / 2),
                    (B / 2,     C, E / 2),
                    (D / 2, E / 2,     F)))

    assert np.linalg.det(A_Q) != 0, 'degenerate conic'
    # As long as det(A_Q) is not 0 it is not degenerate and we can work with the
    # minor 2x2 matrix

    A_33 = np.array(((    A, B / 2),
                     (B / 2,     C)))

    # (det == 0)->parabola, (det < 0)->hyperbola, (det > 0)->ellipse
    assert np.linalg.det(A_33) > 0, 'conic is not an ellipse'

    # Centers are obtained by solving for where the gradient of the quadratic
    # becomes 0. Without going through the derivation the calculation is...
    # These should be 0, 0 if we are at the origin, or our original x, y
    # coordinate specified by the keypoints. I'm doing the calculation just for
    # shits and giggles
    x_center = (B * E - (2 * C * D)) / (4 * A * C - B ** 2)
    y_center = (D * B - (2 * A * E)) / (4 * A * C - B ** 2)

    #=================
    # DRAWING
    #=================
    # Now we are going to determine the major and minor axis
    # of this beast. It just the center augmented by the eigenvecs
    l1, l2, v1, v2 = _2x2_eig(A_33)
    dx1, dy1 = 0 - (v1 / np.sqrt(l1))
    dx2, dy2 = 0 - (v2 / np.sqrt(l2))
    _plotarrow(0, 0, dx1, dy1, color=df2.BLUE)
    _plotarrow(0, 0, dx2, dy2, color=df2.BLUE)

    # The angle between the major axis and our x axis is:
    x_axis = np.array([1, 0])
    theta = np.arccos(x_axis.dot(evec1))


    # The eccentricity is determined by:
    nu = 1
    numer  = 2 * np.sqrt((A - C) ** 2 + B ** 2)
    denom  = nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
    eccentricity = np.sqrt(numer / denom)



    from scipy.special import ellipeinc


    # Algebraic form of connic
    #assert (a * (x ** 2)) + (b * (x * y)) + (c * (y ** 2)) + (d * x) + (e * y) + (f) == 0




    #---------------------

    invA = np.array([[a, 0],
                     [c, d]])

    Ashape = np.linalg.inv(np.array([[a, 0],
                                     [c, d]]))
    Ashape /= np.sqrt(np.linalg.det(Ashape))

    tau = 2 * np.pi
    nSamples = 100
    theta_list = np.linspace(0, tau, nSamples)

    # Create unit circle sample
    cicrle_pts  = np.array([(np.cos(t), np.sin(t)) for t in theta_list])
    circle_hpts = np.hstack([cicrle_pts, np.ones((len(cicrle_pts), 1))])

    # Transform as if the unit cirle was the warped patch
    ashape_pts = Ashape.dot(cicrle_pts.T).T

    inv = np.linalg.inv
    svd = np.linalg.svd
    U, S_, V = svd(Ashape)
    S = np.diag(S_)
    pxl_list3 = invA.dot(cicrle_pts[:, 0:2].T).T
    pxl_list4 = invA.dot(ashape_pts[:, 0:2].T).T
    pxl_list5 = invA.T.dot(cicrle_pts[:, 0:2].T).T
    pxl_list6 = invA.T.dot(ashape_pts[:, 0:2].T).T
    pxl_list7 = inv(V).dot(ashape_pts[:, 0:2].T).T
    pxl_list8 = inv(U).dot(ashape_pts[:, 0:2].T).T
    df2.draw()


    def _plot(data, px, title=''):
        df2.figure(9003, docla=True, pnum=(2, 4, px))
        df2.plot2(data.T[0], data.T[1], '.', title)

    df2.figure(9003, doclf=True)
    _plot(cicrle_pts, 1, 'unit circle')
    _plot(ashape_pts, 2, 'A => circle shape')
    _plot(pxl_list3, 3)
    _plot(pxl_list4, 4)
    _plot(pxl_list5, 5)
    _plot(pxl_list6, 6)
    _plot(pxl_list7, 7)
    _plot(pxl_list8, 8)
    df2.draw()


    invA = np.array([[a, 0, x],
                     [c, d, y],
                     [0, 0, 1]])

    pxl_list = invA.dot(circle_hpts.T).T[:, 0:2]

    df2.figure(9002, doclf=True)
    df2.imshow(rchip)
    df2.plot2(pxl_list.T[0], pxl_list.T[1], '.')
    df2.draw()

    vals = [cv2.getRectSubPix(rchip, (1, 1), tuple(pxl)) for pxl in pxl_list]
    return vals


def get_patch(rchip, kp):
    'Returns cropped unwarped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy = kp2_sf(kp)
    ratio = max(sfx, sfy) / min(sfx, sfy)
    radx = sfx * ratio
    rady = sfy * ratio
    #print('[get_patch] sfy=%r' % sfy)
    #print('[get_patch] sfx=%r' % sfx)
    #print('[get_patch] ratio=%r' % ratio)
    (chip_h, chip_w) = rchip.shape[0:2]
    #print('[get_patch()] chip wh = (%r, %r)' % (chip_w, chip_h))
    #print('[get_patch()] kp = %r ' % kp)
    quantx = quantize_to_pixel_with_offset(x, radx, 0, chip_w)
    quanty = quantize_to_pixel_with_offset(y, rady, 0, chip_h)
    ix1, ix2, xm = quantx
    iy1, iy2, ym = quanty
    patch = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy()  # subkeypoint in patch coordinates
    subkp[0:2] = (xm, ym)
    return patch, subkp


def quantize_to_pixel_with_offset(z, radius, low, high):
    ''' Quantizes a small area into an indexable pixel location
    Returns: pixel_range=(iz1, iz2), subpxl_offset
    Pixels:
    +___+___+___+___+___+___+___+___+
      ^     ^ ^                    ^
      z1    z iz                   z2
            ________________________ < radius
                _____________________ < quantized radius
    ========|
                '''
    #print('quan pxl: z=%r, radius=%r, low=%r, high=%r' % (z, radius, low, high))
    #print('-----------')
    #print('z = %r' % z)
    #print('radius = %r' % radius)
    # Non quantized bounds
    z1 = z - radius
    z2 = z + radius
    #print('bounds = %r, %r' % (z1, z2))
    # Quantized bounds
    iz1 = int(max(np.floor(z1), low))
    iz2 = int(min(np.ceil(z2), high))
    #print('ibounds = %r, %r' % (iz1, iz2))
    # Quantized min radius
    z_radius1 = int(np.ceil(z - iz1))
    z_radius2 = int(np.ceil(iz2 - z))
    z_radius = min(z_radius1, z_radius2)
    #print('z_radius=%r' % z_radius)
    return iz1, iz2, z_radius


def kp2_sf(kp):
    'computes scale factor of keypoint'
    (x, y, a, c, d) = kp
    A = np.array(((a, 0), (c, d)))
    U, S, V = svd(A)
    # sf = np.sqrt(1 / (a * d))
    sfx = S[1]
    sfy = S[0]
    return sfx, sfy
