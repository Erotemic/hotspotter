import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    matplotlib.use('Qt4Agg', warn=True, force=True)
    matplotlib.rcParams['toolbar'] = 'None'
import hotspotter.df2 as df2
import hotspotter.helpers as helpers
import matplotlib.pyplot as plt

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt

def numpy_test():
    # Constants
    inv   = scipy.linalg.inv
    sqrtm = scipy.linalg.sqrtm
    tau = np.pi*2
    sin = np.sin
    cos = np.cos
    # Variables
    N = 2**3
    theta = np.linspace(0, tau, N)
    a, b, c, d = (1, 0, .5, .8)

    xc = np.array((sin(theta), cos(theta))) # points on unit circle 2xN

    A = np.array([(a, b),   # maps points on ellipse
                  (c, d)])  # to points on unit circle

    # Test data
    Ainv = inv(A)

    # Test data
    xe = Ainv.dot(xc)

    # Test Ellipse
    E = A.T.dot(A) # equation of ellipse 
    test = lambda ell: ell.T.dot(E).dot(ell).diagonal()

    print all(np.abs(1 - test(xe)) < 1e-9)

    # Start Plot
    df2.reset()
    def plotell(ell):
        df2.plot(ell[0], ell[1])
    # Plot Circle
    fig = df2.figure(1, plotnum=121, title='xc')
    plotell(xc)
    # Plot Ellipse
    fig = df2.figure(1, plotnum=122, title='xe = inv(A).dot(xc)')
    plotell(xe)
    # End Plot
    df2.set_geometry(fig, 1000, 75, 500, 500)
    df2.update()

# E = ellipse
#
# points x on ellipse satisfy x.T * E * x = 1
#
# A.T * A = E

# A = transforms points on an ellipse to a unit circle
def sqrtm_eq():
    M = np.array([(33, 24), (48, 57)])
    R1 = np.array([(1, 4), (8, 5)])
    R2 = np.array([(5, 2), (4, 7)])
    print M
    print R1.dot(R1)
    print R2.dot(R2)
    '''

    matrix is positive semidefinite 
    if x.conjT.dot(M).dot(x) >= 0

    or if rank(A) = rank(A.conjT.dot(A))
    '''


def sympy_test():
    # https://sympy.googlecode.com/files/sympy-0.7.2.win32.exe
    import sympy
    import sympy.galgebra.GA as GA
    import sympy.galgebra.latex_ex as tex
    import sympy.printing as symprint
    import sympy.abc
    import sympy.mpmath

    a, b, c, d = sympy.symbols('a b c d')
    theta = sympy.abc.theta
    sin = sympy.functions.elementary.trigonometric.sin
    cos = sympy.functions.elementary.trigonometric.cos
    sqrtm = sympy.mpmath.sqrtm

    xc = (sin(theta), cos(theta))
    A = sympy.Matrix([(a, b), (c, d)])
    A = sympy.Matrix([(a, 0), (c, d)])
    Ainv = A.inv()

    print('Inverse of lower triangular [(a, 0), (c, d)]')
    print(Ainv) # sub to lower triangular
    print('--')

    Asqrtm = sqrtm(A)


    print('Inverse of lower triangular [(a, 0), (c, d)]')
    print(Ainv) # sub to lower triangular
    print('--')

    def asMatrix(list_): 
        N = int(len(list_) / 2)
        Z = sympy.Matrix([list_[0:N], list_[N:]])
        return Z

    def simplify_mat(X):
        list_ = [_.simplify() for _ in X]
        Z = asMatrix(list_)
        return Z

    def symdot(X, Y):
        Z = asMatrix(X.dot(Y))
        return Z

    Eq = sympy.Eq
    solve = sympy.solve
    a, b, c, d = sympy.symbols('a b c d')
    w, x, y, z = sympy.symbols('w x y z')
    R = sympy.Matrix([(w, x), (y, z)])
    M = symdot(R,R)
    # Solve in terms of A
    w1 = solve(Eq(a, M[0]), w)[1].subs(y,0)
    #sympy.Eq(0, M[1]) # y is 0
    x1 = solve(Eq(c, M[2]), x)[0]
    z1 = solve(Eq(d, M[3]), z)[1].subs(y,0)
    x2 = x1.subs(w, w1).subs(z, z1)

    R_intermsof_A = sympy.Matrix([(w1, x2), (0, z1)])

    print('R = sqrtm(A) in terms of A: ')
    print(R_intermsof_A)

    print('\nInverse Square Root of Lower Triangular: ')
    print(simplify_mat(R_intermsof_A.inv()))

    A2 = simplify_mat(R_intermsof_A.dot(R_intermsof_A))


    E_ = A.T.dot(A)
    E = sympy.Matrix([E_[0:2], E_[2:4]])
    print('Original ellipse matrix: E = A.T.dot(A)')
    print(E.subs(b,0))

    E_evects = E.eigenvects()
    E_evals = E.eigenvals()

    e1, e2 = E_evects
    print('Eigenvect1: ')
    e1[0].subs(b,0)
    print('Eigenvect2: ')
    e1[0].subs(b,0)


    xe = Ainv.dot(xc)


#A2 = sqrtm(inv(A)).real
#A3 = inv(sqrtm(A)).real
#fig = df2.figure(1, plotnum=223, title='')
#plotell(e2)
#fig = df2.figure(1, plotnum=224, title='')
#plotell(e3)

df2.reset()
real_data()

r'''
str1 = '\n'.join(helpers.execstr_func(sympy_data).split('\n')[0:-2])
str2 = '\n'.join(helpers.execstr_func(execute_test).split('\n')[3:])

toexec = str1 + '\n' + str2
exec(toexec)
'''
