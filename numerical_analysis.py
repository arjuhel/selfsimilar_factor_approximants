# -*- coding: utf-8 -*-

import quadpy
import numpy.linalg as m
import sympy as sym
from sympy.utilities.lambdify import lambdify
import scipy.optimize as scio
from random import random
import numpy as np

'''
#### Numerical integration
Code author: Arnaud Juhel
Published on Github for illustrative purposes
https://github.com/arjuhel/selfsimilar_factor_approximants
'''
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def transpose(mat):
    mato = []
    for i in range(len(mat[0])):
        mato.append([])
        for j in range(len(mat)):
            mato[-1].append(mat[j][i])
    return mato

from functools import reduce
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)
        
'''
    ###### Gaussian quadrature rules from moments ######
    
    Assume a system of non-linear equations that form a Vandermonde system:
    n_1 + n_2 + n_3 + n_4 = B_0
    n_1 * A_1 + n_2 * A_2 + n_3 * A_3 + n_4 * A_4 = B_1
    n_1 * A_1^2 + n_2 * A_2^2 + n_3 * A_3^2 + n_4 * A_4^2 = B_2
    n_1 * A_1^3 + n_2 * A_2^3 + n_3 * A_3^3 + n_4 * A_4^3 = B_3
    n_1 * A_1^4 + n_2 * A_2^4 + n_3 * A_3^4 + n_4 * A_4^4 = B_4
    
    Where we know the moments B_k. We want to solve for the values of $A$ and $n$.
    Solving this system is equivalent to solving a 4-point quadrature rule.
    
    There are at least two methods that can be used to solve this type of system:
    1. 
    Solving the linear equations (gives a polynomial P_k(u) of which the roots are the nodes of the quadrature 
    Knowing the As, then solve for the $n$'s , which is just solving a linear Vandermonde system.
    Implementation of this method is
    
    Gaussian_quadrature_nodes_weights([B_0, B_1, ..., B_k])
    
    
    2. Find the recurrsion coefficients of the three-term recurrence of orthogonal polynomials (Chebyshev's algorithm).
    Then use the Golub-Welsch algorithm to extract the nodes and weights from the recursion coefficients.
    Implementation of this method is:
    
    GaussianQuadrature([B_0, B_1, ..., B_k])
    
    Remark: this method *will fail* if one of the recursion coefficients \beta is negative because the weights are solved
    as an eigenproblem where the input tridiagonal matrix is computed from the square root of the betas.
    
    
'''



def GaussianQuadrature(mmts):
    
    #This procedure recovers the nodes and weights of a Gaussian quadrature via Chebyshev and Golub-Welsch algorithm.
    
    from scipy import sparse
    
    # Chebyshev's algorithm
    # This is a three-term recurrence algorithm
    def Chebyshev_algo(mmts, prec=12): 
        '''        
        In 1859, Chebyshev obtained a method for determining the recursion coefficients ak, bk from the moments m_j 
        for the orthogonal polynomial recurrence:
        $P_{-1}(x) = 0$
        $P_0(x) = 1$
        $P_{k+1}(x) = (x-a_k)*P_k(x) - b_k*P_{k-1}(x)$ 
        '''
        # misc. variables
        n = int(len(mmts) / 2)
        si = mmts
        
        # initialize the quantities:
        ak = [0] * n
        ak[0] = mmts[1]/mmts[0]
        
        bk = [0] * n
        bk[0] = mmts[0]
        
        sp = [0] * (2*n)
        #print("sp", sp)
        for k in range(1, n):
            sp[k - 1] = si[k - 1]
            #print("sp",k-1, sp)
            for j in range(k, 2*n-k):
                v = sp[j]
                sp[j] =  si[j]
                
                s = si[j]
                si[j] = si[j+1] - ak[k - 1] * s - bk[k - 1] * v
                #print("si", k, j, si)
                #print("sp", k, j, sp)
                #print("ak", k, j, ak)
                #print("bk", k, j, bk)
                #print("\n")
                
            ak[k] = si[k + 1]/si[k] - sp[k]/sp[k - 1]
            bk[k] = si[k]/sp[k - 1]

        #return [[ak[k] for k in range(0, n)], [bk[k] for k in range(0, n)]]
        return ak, bk

    def Golub_Welsch(ak, bk):
        '''
        Golub-Welsch algorithm
        Golub and Welsch show that for all integers m ≥ 1 the nodes are the eigenvalues of the symmetric tridiagonal matrix 
        (known as the Jacobi matrix associated with the Gaussian quadrature formula)
        More technically, this algorithm works by observing that the nodes and weights can be obtained by solving the eigenproblem
        for the Jacobi matrix (sometimes noted T) as follows: let U be an orthogonal matrix containing eigenvectors of T, then
        U^T TU = diag(\zeta_j)
        w_j = \mu_0 * u^2_{1,j}
        
        '''
        jacobi_mat = sparse.diags([ak, [pow(x, .5) for x in bk[1:]], [pow(x, .5) for x in bk[1:]]], [0, -1, 1]).toarray()
        #print(jacobi_mat)
        eigvals = np.linalg.eig(jacobi_mat)
    #         print("eigenvalues")
    #         print(eigvals)

        nodes = eigvals[0]
        # and the weights are proportional to the squares
        # of the first components of the normalized eigenvectors
        weights = [bk[0] * pow(ev, 2.) for (i, ev) in enumerate(eigvals[1][0])]
        
        
        return zip(*sorted(zip(nodes, weights), reverse=True))

    
    #Run Chebyshev + Golub-Welsch routine to find weights and nodes of the quadrature rule from the moments
    ak, bk = Chebyshev_algo(mmts)
#     print("\n", "Recursion coefficients")
#     print("ak", ak)
#     print("bk", bk)
    return Golub_Welsch(ak, bk)



def Gaussian_quadrature_nodes_weights(mom, printMatrices=False):

    # Generating Gaussian quadrature rules from moments
    # My Python implementation of this mathematica algo proposed here:
    #https://math.stackexchange.com/questions/13174/solving-a-peculiar-system-of-equations#13197
    #https://math.stackexchange.com/questions/63009/need-help-solving-a-particular-system-of-non-linear-equations-analytically
   
    import cmath
    J=cmath.exp(2j*cmath.pi/3)
    Jc=1/J

    def Ferrari(a,b,c,d,e):
        "Ferrari's Method"
        "resolution of P=ax^4+bx^3+cx^2+dx+e=0, coeffs reals"
        "First shift : x= z-b/4/a  =>  P=z^4+pz^2+qz+r"
        z0=b/4/a
        a2,b2,c2,d2 = a*a,b*b,c*c,d*d
        p = -3*b2/(8*a2)+c/a
        q = b*b2/8/a/a2 - 1/2*b*c/a2 + d/a
        r = -3/256*b2*b2/a2/a2 +c*b2/a2/a/16-b*d/a2/4+e/a
        "Second find y so P2=Ay^3+By^2+Cy+D=0"
        A=8
        B=-4*p
        C=-8*r
        D=4*r*p-q*q
        y0,y1,y2=Cardano(A,B,C,D)
        if abs(y1.imag)<abs(y0.imag): y0=y1
        if abs(y2.imag)<abs(y0.imag): y0=y2
        a0=(-p+2*y0)**.5
        if a0==0 : b0=y0**2-r
        else : b0=-q/2/a0
        r0,r1=Roots_2(1,a0,y0+b0)
        r2,r3=Roots_2(1,-a0,y0-b0)
        return (r0-z0,r1-z0,r2-z0,r3-z0)

    def Cardano(a,b,c,d):
        z0=b/3/a
        a2,b2 = a*a,b*b
        p=-b2/3/a2 +c/a
        q=(b/27*(2*b2/a2-9*c/a)+d)/a
        D=-4*p*p*p-27*q*q
        r=cmath.sqrt(-D/27+0j)
        u=((-q-r)/2)**0.33333333333333333333333
        v=((-q+r)/2)**0.33333333333333333333333
        w=u*v
        w0=abs(w+p/3)
        w1=abs(w*J+p/3)
        w2=abs(w*Jc+p/3)
        if w0<w1:
            if w2<w0 : v*=Jc
        elif w2<w1 : v*=Jc
        else: v*=J
        return u+v-z0, u*J+v*Jc-z0, u*Jc+v*J-z0

    def Roots_2(a,b,c):
        bp=b/2
        delta=bp*bp-a*c
        r1=(-bp-delta**.5)/a
        r2=-r1-b/a
        return r1,r2
    
    def partition(lst, n, d):
        # generates sublists with offset d. 
        # Partition into sublists of length n with offset d:
        rem = len(lst)
        op = []
        i = 0
        while rem > n:
            op.append(lst[i:i+n])
            i += d
            rem -= d
        return op

    def HankelMatrix(fc, fr):
        #gives the Hankel matrix with elements ci down the first column, and ri across the last row.
        assert len(fc) == len(fr)
        #c_m must be the same as r_1
        assert fc[-1] == fr[0]

        mat = transpose([fc])
        mat[-1] = fr
        #we know the anti diagonal = fr[0]
        for i in range(len(fc)-2, -1, -1):
            for j in range(1, len(fc)):
                mat[i].append(mat[i+1][j-1])

        return mat

    def FoldList(f, x, lst):
        op = [x]
        i = 0
        while i < len(lst):
            
            op.append(sym.expand(f(op[-1], lst[i])))
            i += 1

        return op

    def Fold(f, x, lst): # python translation of the Mathematica Fold function
        return FoldList(f, 1, lst)[-1]
    
    def GaussianPolynomial(mmts):
        # outputs a Gaussian polynomial
        # mmts is a vector of moments
        # upper bound for the nsolve
        n = int(len(mmts) / 2)
        p = partition(mmts, n, n-1)
       
        if len(p) < 2:
            print("Partition is not correct:", p)
        t = [-e for e in mmts[-n:]]
        hm = HankelMatrix(*p[:2])
        s = m.solve(hm, t)[::-1]
        
        if printMatrices:
            print("Partition", p)
            print("Hankel matrix")
            print_Table_V(transpose(hm) + transpose([t]), 2)
            
            print("Solution", s)
        
        t = sym.symbols("t")
        x1, x2 = sym.symbols("x1 x2")
        f = lambda x1, x2: x1 * t + x2
        poly = Fold(f, 1, s)
        
        return poly


    
    def VandermondeSolve(lst, n, y):
        #print(lst)
        vmat = [[pow(lst[i], j) for i in range(0, len(lst))]  for j in range(1, len(y)+1)]
        if printMatrices:
            print("Vandermonde augmented system")
            print_Table_V(transpose(transpose(vmat) + [y]), 8)
            
#         sbls = "n1 "
#         for i in range(1, len(lst)):
#             sbls += "n" + str(i+1) + " "
#         symbols = sym.symbols(sbls)
#         return sym.linsolve((vmat, [[0]]*len(lst)), symbols)

        def f(x):
            z = np.dot(vmat, x) - y
            return np.dot(z, z)
        
        res = scio.minimize(f, [0.5]*len(lst), method='SLSQP')
        
        if not res['success'] == True:
            print(res)
            raise Exception()
        
        return list(res["x"])  
    ###END OF VANDERMONDESOLVE

    
    
    
    
    #### ROUTINE
    ln = len(mom)
    n = int(ln/2)
    poly = GaussianPolynomial(mom)
    t = sym.symbols("t")
    coefs = [float(poly.coeff(t, i)) for i in reversed(range(n+1))]

        
    assert len(coefs) == n+1

    if n == 4:
        #First find the roots of the quartic function using Ferrari's method
        roots = list(np.real(Ferrari(*coefs)))
        
        # All the nodes must be positive for later use, so if (at least one) roots are negative,
        # have to find another set of values which minimize error to the moments
        neg_rts = False
        for i in range(0, n):
            neg_rts = neg_rts or roots[i] < 0 
        qf = sym.lambdify(t, poly)
        
        # objective: find four root-like positive values of which return a unique result closest to 0.

        nodes = roots
        
    elif n == 2 or n == 3:
        #print(coefs)
        #roots = Cardano(*coefs)
        roots = list(sym.solve([poly, t>0], t, 1).atoms())
        roots = [float(item) for item in roots if isinstance(item, sym.numbers.Float)]
        #print([type(i) for i in roots])

        if not len(roots) == n:
            raise Exception(roots)#check that the number of roots is correct

        nodes = roots
    else:
        raise Exception("Not supported.")
        
    try:
        nodes.sort(reverse=True)
    except:
        raise Exception(nodes)
    
    #Solve the final linear system to find the weights
    weights = VandermondeSolve(nodes, len(nodes), mom)
    
    return nodes, weights


def reverse_Chebyshev(ak, bk):
        
    import sympy as sym
    '''
    Extracting moments given Chebyshev's recursion coefficients.
    To find what moments they used in the paper, 
    I made this inverse Chebyshev method on the observation that
    if we run Chebyshev's algorithm symbolically, we obtain the following equations:

    b1 = m_1
    a1 = m_2/m_1
    b2 = (m_1*m_3 - m_2**2)/m_1**2
    a2 = (m_1**2*m_4 - 2*m_1*m_2*m_3 + m_2**3)/(m_1*(m_1*m_3 - m_2**2))
    b3 = m_1*(m_1*m_3*m_5 - m_1*m_4**2 - m_2**2*m_5 + 2*m_2*m_3*m_4 - m_3**3)/(m_1**2*m_3**2 - 2*m_1*m_2**2*m_3 + m_2**4)
    a3 = (-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1)
    b4 = (m_1**2*m_3**2*m_5*m_7 - m_1**2*m_3**2*m_6**2 - m_1**2*m_3*m_4**2*m_7 + 2*m_1**2*m_3*m_4*m_5*m_6 - m_1**2*m_3*m_5**3 - 2*m_1*m_2**2*m_3*m_5*m_7 + 2*m_1*m_2**2*m_3*m_6**2 + m_1*m_2**2*m_4**2*m_7 - 2*m_1*m_2**2*m_4*m_5*m_6 + m_1*m_2**2*m_5**3 + 2*m_1*m_2*m_3**2*m_4*m_7 - 2*m_1*m_2*m_3**2*m_5*m_6 - 2*m_1*m_2*m_3*m_4**2*m_6 + 2*m_1*m_2*m_3*m_4*m_5**2 - m_1*m_3**4*m_7 + 2*m_1*m_3**3*m_4*m_6 + m_1*m_3**3*m_5**2 - 3*m_1*m_3**2*m_4**2*m_5 + m_1*m_3*m_4**4 + m_2**4*m_5*m_7 - m_2**4*m_6**2 - 2*m_2**3*m_3*m_4*m_7 + 2*m_2**3*m_3*m_5*m_6 + 2*m_2**3*m_4**2*m_6 - 2*m_2**3*m_4*m_5**2 + m_2**2*m_3**3*m_7 - 2*m_2**2*m_3**2*m_4*m_6 - m_2**2*m_3**2*m_5**2 + 3*m_2**2*m_3*m_4**2*m_5 - m_2**2*m_4**4)/(m_1**2*m_3**2*m_5**2 - 2*m_1**2*m_3*m_4**2*m_5 + m_1**2*m_4**4 - 2*m_1*m_2**2*m_3*m_5**2 + 2*m_1*m_2**2*m_4**2*m_5 + 4*m_1*m_2*m_3**2*m_4*m_5 - 4*m_1*m_2*m_3*m_4**3 - 2*m_1*m_3**4*m_5 + 2*m_1*m_3**3*m_4**2 + m_2**4*m_5**2 - 4*m_2**3*m_3*m_4*m_5 + 2*m_2**2*m_3**3*m_5 + 4*m_2**2*m_3**2*m_4**2 - 4*m_2*m_3**4*m_4 + m_3**6)
    a4 = (m_8 - (m_7 - m_2*m_6/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + ((-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1))*(-m_7 + (m_6 - m_2*m_5/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_6/m_1 + m_5*(m_3 - m_2**2/m_1)/m_1) + (m_6 - m_2*m_5/m_1)*(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1)/(m_3 - m_2**2/m_1) - m_2*m_7/m_1 - m_6*(m_3 - m_2**2/m_1)/m_1)/(m_7 - (m_6 - m_2*m_5/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + ((-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1))*(-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1) + (m_5 - m_2*m_4/m_1)*(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1)/(m_3 - m_2**2/m_1) - m_2*m_6/m_1 - m_5*(m_3 - m_2**2/m_1)/m_1) - (-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1)

    We observe that each subsequent equation contains one additional unknown. 
    So we just solve and substitute into the next equation, then reorder and solve to obtain the moments.
    Current implementation works for up to 8 coefficients.

    '''


    def mom_solve(lhs, rhs, tgt, *args):
        return sym.solve(lhs-rhs, tgt)

    n = len(ak) + len(bk)
    # Define what we can know from the coefficients
    #a1, a2, a3, a4, b1, b2, b3, b4 = sym.symbols("a_1 a_2 a_3 a_4 b_1 b_2 b_3 b_4")
    a1, a2, a3, a4 = ak
    b1, b2, b3, b4 = bk
    m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8 = sym.symbols("m_1 m_2 m_3 m_4 m_5 m_6 m_7 m_8")

    m_1 = b1
    m_2 = a1 * b1
    moms = [m_1, m_2]
    m_3 = mom_solve(b2, (m_1*m_3 - m_2**2)/m_1**2, m_3, moms)[0]
    moms.append(m_3)
    m_4 = mom_solve(a2, (m_1**2*m_4 - 2*m_1*m_2*m_3 + m_2**3)/(m_1*(m_1*m_3 - m_2**2)), m_4, moms)[0]
    moms.append(m_4)
    m_5 = mom_solve(b3, m_1*(m_1*m_3*m_5 - m_1*m_4**2 - m_2**2*m_5 + 2*m_2*m_3*m_4 - m_3**3)/(m_1**2*m_3**2 - 2*m_1*m_2**2*m_3 + m_2**4), m_5, moms)[0]
    moms.append(m_5)
    m_6 = mom_solve(a3, (-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1), m_6, moms)[0]
    moms.append(m_6)
    m_7 = mom_solve(b4, (m_1**2*m_3**2*m_5*m_7 - m_1**2*m_3**2*m_6**2 - m_1**2*m_3*m_4**2*m_7 + 2*m_1**2*m_3*m_4*m_5*m_6 - m_1**2*m_3*m_5**3 - 2*m_1*m_2**2*m_3*m_5*m_7 + 2*m_1*m_2**2*m_3*m_6**2 + m_1*m_2**2*m_4**2*m_7 - 2*m_1*m_2**2*m_4*m_5*m_6 + m_1*m_2**2*m_5**3 + 2*m_1*m_2*m_3**2*m_4*m_7 - 2*m_1*m_2*m_3**2*m_5*m_6 - 2*m_1*m_2*m_3*m_4**2*m_6 + 2*m_1*m_2*m_3*m_4*m_5**2 - m_1*m_3**4*m_7 + 2*m_1*m_3**3*m_4*m_6 + m_1*m_3**3*m_5**2 - 3*m_1*m_3**2*m_4**2*m_5 + m_1*m_3*m_4**4 + m_2**4*m_5*m_7 - m_2**4*m_6**2 - 2*m_2**3*m_3*m_4*m_7 + 2*m_2**3*m_3*m_5*m_6 + 2*m_2**3*m_4**2*m_6 - 2*m_2**3*m_4*m_5**2 + m_2**2*m_3**3*m_7 - 2*m_2**2*m_3**2*m_4*m_6 - m_2**2*m_3**2*m_5**2 + 3*m_2**2*m_3*m_4**2*m_5 - m_2**2*m_4**4)/(m_1**2*m_3**2*m_5**2 - 2*m_1**2*m_3*m_4**2*m_5 + m_1**2*m_4**4 - 2*m_1*m_2**2*m_3*m_5**2 + 2*m_1*m_2**2*m_4**2*m_5 + 4*m_1*m_2*m_3**2*m_4*m_5 - 4*m_1*m_2*m_3*m_4**3 - 2*m_1*m_3**4*m_5 + 2*m_1*m_3**3*m_4**2 + m_2**4*m_5**2 - 4*m_2**3*m_3*m_4*m_5 + 2*m_2**2*m_3**3*m_5 + 4*m_2**2*m_3**2*m_4**2 - 4*m_2*m_3**4*m_4 + m_3**6), m_7, moms)[0]
    moms.append(m_7)
    m_8 = mom_solve(a4, (m_8 - (m_7 - m_2*m_6/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + ((-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1))*(-m_7 + (m_6 - m_2*m_5/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_6/m_1 + m_5*(m_3 - m_2**2/m_1)/m_1) + (m_6 - m_2*m_5/m_1)*(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1)/(m_3 - m_2**2/m_1) - m_2*m_7/m_1 - m_6*(m_3 - m_2**2/m_1)/m_1)/(m_7 - (m_6 - m_2*m_5/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + ((-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1) - (m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1))*(-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1) + (m_5 - m_2*m_4/m_1)*(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1)/(m_3 - m_2**2/m_1) - m_2*m_6/m_1 - m_5*(m_3 - m_2**2/m_1)/m_1) - (-m_6 + (m_5 - m_2*m_4/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_5/m_1 + m_4*(m_3 - m_2**2/m_1)/m_1)/(-m_5 + (m_4 - m_2*m_3/m_1)*((m_4 - m_2*m_3/m_1)/(m_3 - m_2**2/m_1) - m_2/m_1) + m_2*m_4/m_1 + m_3*(m_3 - m_2**2/m_1)/m_1), m_8, moms)[0]
    moms.append(m_8)

    return [float(m) for m in moms]

    
# Example taken from Yukalov, Gluzman and Sornette (2003)
alphaT, betaT = quadpy.tools.coefficients_from_gauss([2,1,.5,.1], [1.5,.5,1/3,1/4])
#print("\n", "Theo. coeffs from ground truth")
# print("ak", list(alphaT))
# print("bk", list(betaT), "\n")
paper_moms = reverse_Chebyshev(alphaT, betaT)
# print("Moments used in paper")
# print(paper_moms)

## Unit test for Gaussian quadrature via Chebyshev-Golub-Welsch
A4p, n4p = GaussianQuadrature(paper_moms)
#print_Table_V([A4p, n4p], 12)
assert isclose(2., A4p[0], 10) and isclose(1., A4p[1], 10) and isclose(0.5, A4p[2], 10) and isclose(0.1, A4p[3], 10)
assert isclose(1.5, n4p[0], 10) and isclose(0.5, n4p[1], 10) and isclose(0.25, n4p[2], 10) and isclose(1/3, n4p[3], 10)

A4p, n4p = Gaussian_quadrature_nodes_weights(paper_moms)
#print_Table_V([A4p, n4p], 12)
assert isclose(2., A4p[0], 10) and isclose(1., A4p[1], 10) and isclose(0.5, A4p[2], 10) and isclose(0.1, A4p[3], 10)
assert isclose(1.5, n4p[0], 10) and isclose(0.5, n4p[1], 10) and isclose(0.25, n4p[2], 10) and isclose(1/3, n4p[3], 10)
