
#pip install pexpect
#sympy.init_printing(use_unicode=True)

def view(expr):
    symprint.preview(expr, output='png')

w, x, y, z = sympy.symbols('w x y z')
tex.LaTeX(theta)
x = sympy.symbols('x')
# Rotation matrix
Q = sympy.Matrix([(cos(theta), -sin(theta)), (sin(theta), cos(theta))])
tex.LaTeX(Q)
A = A.subs(b, 0)

E2 = A.T * A

# Unit circle to ellipse
E = sympy.Matrix([(w, x), (y, z)])

E2 = A.T * A

E3 = (Q*A).T * (Q * A)

A.inv("LU")

L_A, U_A, row_swaps = A.LUdecomposition()
a, b, c, d = sympy.symbols('a b c d')
w, x, y, z = sympy.symbols('w x y z')

#b = 0

theta = sympy.abc.theta
tex.LaTeX(theta)

sin = sympy.functions.elementary.trigonometric.sin
cos = sympy.functions.elementary.trigonometric.cos

# Rotation matrix
Q = sympy.Matrix([(cos(theta), -sin(theta)), (sin(theta), cos(theta))])
tex.LaTeX(Q)

# From ellipse to unit circle
A = sympy.Matrix([(a, b), (c, d)])

# Unit circle to ellipse
E = sympy.Matrix([(w, x), (y, z)])

E2 = A.T * A

E3 = (Q*A).T * (Q * A)

A.inv("LU")

L_A, U_A, row_swaps_A = A.LUdecomposition()
L_Q, U_Q, row_swaps_Q = Q.LUdecomposition()

print A
print E2
print E3

tex.Format()

tex.LaTeX(theta)
e2_tex = tex.LaTeX(E2)

symprint.preview(E2)

sympy.Eq(A.transpose().dot(A), E)


(xbm, alpha_1, delta__nugamma_r)  = GA.make_symbols('xbm alpha_1 delta__nugamma_r')
x = alpha_1*xbm/delta__nugamma_r
print 'x =',x
tex.LaTeX(x)
tex.xdvi()
 #----
from sympy import Matrix

A = Matrix([ [2, 3, 5], [3, 6, 2], [8, 3, 6] ])
x = Matrix(3,1,[3,7,5])
