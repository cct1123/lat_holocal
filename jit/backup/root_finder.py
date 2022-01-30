'''
Some methods to find the root of a function. which are all local seraching methods meaning 
that users have to specify a valid range in which the root must lie. The methods are 
compatible with JIT of numba, show computational advantages over those in scipy.optimize
only when they are used repeatedly like the use case in ray tracing.

Chun Tung Cheung (January 2022)
'''

import numpy as np
import numba as nb
from numba import jit
@jit(nopython=True, fastmath=True)
def bisection(func, a, b, tol=1e-12, iter_nmax=100):
    '''
    Bisection method is the most robust root finder method,
    but converges slowly for highly non-linear functions.
    Always use this method when the function is a simple polynomial (e.g. linear, quadratic)

    Arguemtns :
        func : a python function compatible with numba
        a : the lower bound of the search range
        b : the upper bound of the search range
        tol : tolerance, return the root when abs(a-b)<tol
        iter_nmax : max number of iteration
    Returns :
        the best estimate of the root
    '''
    fa = func(a)
    fb = func(b)
    if fa*fb >= 0: 
        raise ValueError("The root is not bracketed since f(a) and f(b) have the same sign.")

    for i in nb.prange(iter_nmax):
        c = (a + b) / 2
        fc = func(c)
        # print("Iteration %i--------"%i)
        # print("a, fa : %f, %f"%(a, fa))
        # print("b, fb : %f, %f"%(b, fb))
        # print("c, fc : %f, %f"%(c, fc))
        if fc == 0 or np.abs(b-a)/2 < tol:
            # root found
            return c
        if fc*fa>0:
            a = c
        else:
            b = c
        fa = func(a)
        fb = func(b)
    raise Exception("Root non-found. Exceeded max. number of iteration.")

@jit(nopython=True, fastmath=True)
def bisection_arg(func, a, b, funcarg, tol=1e-12, iter_nmax=100):
    '''
    Bisection method is the most robust root finder method,
    but converges slowly for highly non-linear functions.
    Always use this method when the function is a simple polynomial (e.g. linear, quadratic)

    Arguemtns :
        func : a python function compatible with numba
        funcarg : a tuple of parameters for func
        a : the lower bound of the search range
        b : the upper bound of the search range
        tol : tolerance, return the root when abs(a-b)<tol
        iter_nmax : max number of iteration
    Returns :
        the best estimate of the root
    '''
    fa = func(a, *funcarg)
    fb = func(b, *funcarg)
    if fa*fb >= 0: 
        raise ValueError("The root is not bracketed since f(a) and f(b) have the same sign.")

    for i in nb.prange(iter_nmax):
        c = (a + b) / 2
        fc = func(c, *funcarg)
        # print("Iteration %i--------"%i)
        # print("a, fa : %f, %f"%(a, fa))
        # print("b, fb : %f, %f"%(b, fb))
        # print("c, fc : %f, %f"%(c, fc))
        if fc == 0 or np.abs(b-a)/2 < tol:
            # root found
            return c
        if fc*fa>0:
            a = c
        else:
            b = c
        fa = func(a, *funcarg)
        fb = func(b, *funcarg)
    raise Exception("Root non-found. Exceeded max. number of iteration.")

@jit(nopython=True, fastmath=True)
def brentq(func, a, b, tol=1e-12, iter_nmax=100):
    '''
    Brent's method is efficient and yet robust, which combines all the root seraching 
    methods including bisection, secant and inverse quadratice interpolation.

    Arguemtns :
        func : a python function compatible with numba
        a : the lower bound of the search range
        b : the upper bound of the search range
        tol : tolerance, return the root when abs(a-b)<tol
        iter_nmax : max number of iteration
    Returns :
        the best estimate of the root

    see 
    https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/
    '''
    fa = func(a)
    fb = func(b)
    if fa*fb >= 0: 
        raise ValueError("The root is not bracketed since f(a) and f(b) have the same sign.")
    if np.abs(fa)<np.abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    mflag = True
    c = a # initialize c
    d = c # initialize d

    iter_n = 0
    while iter_n < iter_nmax:
        if fb == 0 or np.abs(b-a)/2.0 < tol:
            return b

        fc = func(c)
        # print(f"Iteration {iter_n}--------")
        # print(f"a, fa : {a}, {fa}")
        # print(f"b, fb : {b}, {fb}")
        # print(f"c, fc : {c}, {fc}")

        if fa != fc and fb != fc:
            # inverse quadratice interpolation
            s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb))
        else:
            # secant method
            s = b - fb*(b-a)/(fb-fa)

        cond1 = s < (3*a+b)/4 or s > b
        cond2 = mflag == True and np.abs(s-b) >= np.abs(b-c)/2
        cond3 = mflag == False and np.abs(s-b) >= np.abs(c-d)/2
        cond4 = mflag == True and np.abs(b-c) < tol
        cond5 = mflag == False and np.abs(c-d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a+b)/2
            mflag = True
        else:
            mflag =False

        fs = func(s)

        d = c
        c = b
        if fa*fs > 0:
            a = s
        else:
            b = s

        if np.abs(fa)<np.abs(fb):
            a, b = b, a
        fa = func(a)
        fb = func(b)

        iter_n += 1
    else:
        raise Exception("Root non-found. Exceeded max. number of iteration.")

@jit(nopython=True, fastmath=True)
def brentq_arg(func, a, b, funcarg, tol=1e-12, iter_nmax=100):
    '''
    Brent's method is efficient and yet robust, which combines all the root seraching 
    methods including bisection, secant and inverse quadratice interpolation.

    Arguemtns :
        func : a python function compatible with numba
        funcarg : a tuple of parameters for func
        a : the lower bound of the search range
        b : the upper bound of the search range
        tol : tolerance, return the root when abs(a-b)<tol
        iter_nmax : max number of iteration
    Returns :
        the best estimate of the root

    see 
    https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/
    '''
    fa = func(a, *funcarg)
    fb = func(b, *funcarg)
    if fa*fb >= 0: 
        raise ValueError("The root is not bracketed since f(a) and f(b) have the same sign.")
    if np.abs(fa)<np.abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    mflag = True
    c = a # initialize c
    d = c # initialize d

    iter_n = 0
    while iter_n < iter_nmax:
        if fb == 0 or np.abs(b-a)/2.0 < tol:
            return b

        fc = func(c, *funcarg)
        # print(f"Iteration {iter_n}--------")
        # print(f"a, fa : {a}, {fa}")
        # print(f"b, fb : {b}, {fb}")
        # print(f"c, fc : {c}, {fc}")

        if fa != fc and fb != fc:
            # inverse quadratice interpolation
            s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb))
        else:
            # secant method
            s = b - fb*(b-a)/(fb-fa)

        cond1 = s < (3*a+b)/4 or s > b
        cond2 = mflag == True and np.abs(s-b) >= np.abs(b-c)/2
        cond3 = mflag == False and np.abs(s-b) >= np.abs(c-d)/2
        cond4 = mflag == True and np.abs(b-c) < tol
        cond5 = mflag == False and np.abs(c-d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a+b)/2
            mflag = True
        else:
            mflag =False

        fs = func(s, *funcarg)

        d = c
        c = b
        if fa*fs > 0:
            a = s
        else:
            b = s

        if np.abs(fa)<np.abs(fb):
            a, b = b, a
        fa = func(a, *funcarg)
        fb = func(b, *funcarg)

        iter_n += 1
    else:
        raise Exception("Root non-found. Exceeded max. number of iteration.")


if __name__ == "__main__":

    '''For code testing '''
    from time import time
    import matplotlib.pyplot as plt

    #### set a root function---------------------------------------------
    @jit(nopython=True, fastmath=True)
    def func(t):
        func = t**3*(np.sin(t))**2+1
        return func
    # x = np.linspace(-3, 6, 200)
    # y = func(x)
    # plt.plot(x, y)
    # plt.show()
    #### test bisection function---------------------------------------------
    loop_num = 10000000
    @jit(nopython=True, parallel=True, fastmath=True)
    def loop_bisection(arg):
        result = 0
        for i in nb.prange(loop_num):
            rn = np.random.random()
            result += arg*bisection(func, -2.2, 5)
        result = result / loop_num
        return result

    t1 = time()
    # result = bisection(func, -2.2, 5)
    result = loop_bisection(1.0)
    t2 = time()
    print(f'Function {bisection.__name__!r} (looped {loop_num} times) executed in {(t2-t1):.8f}s')
    print(f'result: {result}')

    ### test brentq function---------------------------------------------
    loop_num = 10000000
    @jit(nopython=True, parallel=True, fastmath=True)
    def loop_brentq(arg):
        result = 0
        for i in nb.prange(loop_num):
            rn = np.random.random()
            result += arg*brentq(func, -2.2, 5)
        result = result / loop_num
        return result
    t1 = time()
    # result = brentq(func, -2.2, 5)
    result = loop_brentq(1)
    t2 = time()
    print(f'Function {brentq.__name__!r} (looped {loop_num} times) executed in {(t2-t1):.8f}s')
    print(f'result: {result}')

    ### test brentq_arg function---------------------------------------------
    @jit(nopython=True, fastmath=True)
    def func(t, v, u):
        func = t**3*(np.sin(t))**2+v*u
        return func

    loop_num = 10000000
    @jit(nopython=True, parallel=True, fastmath=True)
    def loop_brentq(arg):
        result = 0
        for i in nb.prange(loop_num):
            rn = np.random.random()
            result += arg*brentq_arg(func, -2.2, 5, (1.0, 1.0))
        result = result / loop_num
        return result

    t1 = time()
    # result = brentq_arg(func, -2.2, 5, (1.1, 0.998))
    result = loop_brentq(1)
    t2 = time()
    print(f'Function {brentq.__name__!r} (looped {loop_num} times) executed in {(t2-t1):.8f}s')
    print(f'result: {result}')