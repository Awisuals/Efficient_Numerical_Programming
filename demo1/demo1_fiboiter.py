def fibonacci_iter(n):
    i, j  = 0, 1 # sequence starts with 0, 1 
    while n > 0:
        i, j, n = j, i+j, n-1  
    return i

if __name__=='__main__':
    for n in range(51):
        print(n,fibonacci_iter(n))
    
