'''
Cache decorator from functools (Python 3) and performance profiling
Fibonacci number recursive computation as a test  
'''
import functools
import cProfile


class Fun:
    # no cache
    def fib0(n):
        if n < 2:
            return n
        return Fun.fib0(n-1) + Fun.fib0(n-2)
    
    # self-made cache
    def fib1(n,*,_cache={}):  # local _cache dictionary used here
        if n in _cache:
            return _cache[n]
        if n < 2:
            return n
        result = Fun.fib1(n-1) + Fun.fib1(n-2)
        _cache[n] = result           # Store result in _cache
        return result
       
    # cache decorator
    @functools.lru_cache(maxsize=None)
    def fib2(n):
        if n < 2:
            return n
        return Fun.fib2(n-1) + Fun.fib2(n-2)

if __name__ == '__main__':
    def call(vers,n):
        f = getattr(Fun, 'fib' + str(vers)) # f is Fun.fibvers
        return f(n)
    
    n = 25  # not too large, please :^)

    pr = cProfile.Profile()
    pr.enable()    
    for vers in range(0,3):
        print("call function version ",vers)        
        call(vers,n)
    pr.disable()
    pr.print_stats(sort='time')

