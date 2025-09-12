r'''
Copyright (c) 2025 Antero Voutilainen
Created: 10 09 2025

Description: Finds funtion root from first exercecise.

Example:
    python .\AV_demo2_findroot_iter.py -a 1.0 -b 2.0
'''
import math as m
import argparse as ap


def f(x):
    """Defines function to find root from.

    Args:
        x (foat): x-axle value

    Returns:
        float: function value at x.
    """    
    return x**3*m.sin(x)*m.cos(x)*m.exp(-x)


def bisection(a, b, f=f, tol=1e-6, max_iter=100):
    """Bisection iteration method.

    Args:
        a (float): Start point of iteration interval.
        b (float): End point of iteration interval.
        f (function, optional): Function to find root from. Defaults to f.
        tol (float, optional): Toleranve for iteration. Defaults to 1e-6.
        max_iter (int, optional): Max amount of iterations done. Defaults to 100.

    Raises:
        ValueError: When given interval doesn't have a sign change.

    Returns:
        float: Root of the funcition at given interval.
    """    
    fa, fb = f(a), f(b)
    # Check if interval end points include roots
    if fa == 0: return a
    if fb == 0: return b
    # Check validity of interval
    if fa * fb > 0:
        raise ValueError("No sign change on interval [a, b]. Select new values.")
    # bisection iterator
    for _ in range(max_iter):
        # Calculate midpoint in interval and function at that point
        m = (a+b)/2
        fm = f(m)
        # Check breakout rule
        if abs(fm) <= tol or (b-a)/2 <= tol:
            return m
        # Identify which half has root
        if fa * fm >= 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return (a+b)/2


def main(args):
    """Main function to run script from.

    Args:
        args (argparse.Namespace): CLI arguments expected.

    Raises:
        ValueError: Asks to provide interval points from CLI 
                    if not in debug mode.
    """    
    a = args.a
    b = args.b
    tol = args.tol
    DEBUG = args.debug
    
    if DEBUG:    
        a, b = 1.0, 2.0
        root = bisection(a, b)
        print(f"Your root in given interval of [{a}, {b}] is: {root}")
    
    else:
        if a is None or b is None:
            raise ValueError("Provide -a and -b, when in not debug mode.")
        root = bisection(a,b,f,tol)
        print(f"Your root in given interval of [{a}, {b}] is: {root}")


if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description="Iterates function root")
    parser.add_argument("-a", type=float, help="Interval start point")
    parser.add_argument("-b", type=float, help="Interval end point")
    parser.add_argument("-t", "--tol", type=float, default=1e-6, 
                        help="Tolerance")
    parser.add_argument("-mi", "--max_iter", type=int, default=100, 
                        help="Max number of iterations before stopping.")
    parser.add_argument("-de", "--debug", action="store_true", 
                        help="Activate debug mode, running script using code-defined values.")
    
    args = parser.parse_args()    
    main(args)

