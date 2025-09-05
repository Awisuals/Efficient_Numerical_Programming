import numpy as np
import scipy as sc
import math
from scipy.special import factorial as scfact

"""
From recursion to iteration
===========================
Tail Call Optimization (TCO) is not an integral part of Python, hence the extra work

General info about tail calls:
https://en.wikipedia.org/wiki/Tail_call

C example:
  https://stackoverflow.com/questions/310974/what-is-tail-call-optimization

Python example used below:
  http://blog.moertel.com/posts/2013-05-11-recursive-to-iterative.html
"""

# recursive
# the last step is not just a function call => this is not tail-call optimizable code
def factorial_recursive(n):
    if n < 2: return 1
    return n * factorial_recursive(n-1) 


# convert to a tail call
# still recursive, but the last step is just a function call => this is tail-call optimizable code

def factorial_tail_call(n,acc=1):
    if n < 2: return acc
    return factorial_tail_call(n-1, acc*n)


# factorial_tail_call only modifies the arguments,
# in comes (n,acc) and out goes (n-1,acc*n).
# In other words, the function does *iteration*  (n,acc)=(n-1,acc*n)


# convert to iterative
def factorial_iterative(n,*,acc=1):
    while n > 1: (n, acc) = (n-1, acc*n)
    return acc


if __name__=='__main__':

    # testing scipy factorial
    print('many factorials at once, scipy.special.factorial using NumPy array input:')
    xs = np.array(np.arange(10))
    facs = scfact(xs)
    for x, fac in zip(xs,facs):
        print(f'{x}! = ',fac)

    print('Version information:')
   
    print('numpy version:',np.version.version)
    print('scipy version:',sc.version.version)
    

    # testing self-made and builf in functions
    print(80*'-')
    num = 1000
    print(f'Demo 1 task 2: {num}! computed using five different ways')
    
    # With num = 1000, A) and B) fails and F) gived inf.
    
    print('A) recursive:')
    try:
        print(factorial_recursive(num))
    except:
        print('failed')        
    print('B) tail call recursive:')
    try:
        print(factorial_tail_call(num))
    except:
        print('failed')
    print('C) tail call iterative:')
    try:
        print(factorial_iterative(num))
    except:
         print('failed')
    print('E) math.factorial:')
    try:
        print(math.factorial(num))
    except:
        print('failed')
    print('F)scipy.special.factorial:')
    try:
        print(scfact(num))
    except:
         print('failed')
         