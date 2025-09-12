r'''
Copyright (c) 2025 Antero Voutilainen
Created: 11 09 2025

Description: Time numpy and python sum.

Simply run with:
    python .\AV_demo2_timesums.py
'''
import numpy as np
import time


def time_operation(A, c=1):
    """Prints used time from selected method.

    Args:
        A (array): Array containing summable numbers.
        c (int, optional): numpy or python. Defaults to 1 (numpy).
    """    
    if c == 1:
        start = time.time()
        sum_value = np.sum(A)
        end = time.time()
        print(f"Sum using numpy: {sum_value}")
        print(f"Summing time using numpy: {end - start}")
    
    if c == 2:
        start = time.time()
        sum_value = sum(A)
        end = time.time()
        print(f"Sum using python: {sum_value}")
        print(f"Summing time using python: {end - start}")

    return


def main():

    N = 10000000
    Array = np.random.random(N)

    time_operation(Array, 1)
    time_operation(Array, 2)
    
    return


if __name__ == '__main__':
    main()

