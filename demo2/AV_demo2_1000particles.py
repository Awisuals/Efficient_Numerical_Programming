'''
Copyright (c) 2025 Antero Voutilainen
Created: 11 09 2025

Description: 
'''
import numpy as np
from numpy import linalg as LA
import time
from numba import njit 
import argparse as ap
import scipy as sc
from scipy.spatial.distance import pdist

METHODS = {}
def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def maybe_timed(fn):
    def wrap(*args, timed=False, **kwargs):
        if not timed:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return out, dt
    return wrap


def generate_particles(scale, N=1000):
    
    N=1000
    A_x = np.random.random(N)*scale
    A_y = np.random.random(N)*scale
    A_z = np.random.random(N)*scale
    
    # x = np.array([A_x, A_y, A_z]) # Shape (3, N) 
    x = np.column_stack((A_x, A_y, A_z)) # Shape (N, 3)
    # print(x)   
    return x


@register("norm")
def distances_norm(points):
    distance_values = np.empty(max(points.shape))
    vec_norm = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    for i in range(max(points.shape)):
        x_i, y_i, z_i = points[i]
        dist_i = vec_norm(x_i, y_i, z_i)
        distance_values[i] = dist_i
    return distance_values


@register("norm-numba")
@njit
def distances_norm_numba(points):
    distance_values = np.empty(max(points.shape))
    vec_norm = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    for i in range(max(points.shape)):
        x_i, y_i, z_i = points[i]
        dist_i = vec_norm(x_i, y_i, z_i)
        distance_values[i] = dist_i
    return distance_values


@register("npnorm")
def distances_npnorm(points):
    return LA.norm(points, axis=1)


@register("einsum")
def distances_einsum():
    return np.sqrt(np.einsum('ijk,ijk->ij',d,d))


@register("pdist")
def distances_pdist():
    return


def main():
    
    N_points = 1000
    scaling_value = 5.0
    points = generate_particles(scaling_value, N_points)
    # distance_values = distances_norm(points)
    distance_values = distances_npnorm(points)
    print(distance_values)
    
    return


if __name__ == '__main__':
    main()
    




