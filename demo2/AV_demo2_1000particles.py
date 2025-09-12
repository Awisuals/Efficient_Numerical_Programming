r'''
Copyright (c) 2025 Antero Voutilainen
Created: 11 09 2025

Description: Generates N points with random distance from origo scaled
by given float. Then uses choosed method to compute each points distance
from said origo. Runnable from CLI.

Here's a couple example running commands:
    python .\AV_demo2_1000particles.py -m norm-numba -n 100000000 -s 100.0 -pd
    python .\AV_demo2_1000particles.py -m all -pd
    python .\AV_demo2_1000particles.py -m norm einsum
'''
import numpy as np
from numpy import linalg as LA
import time
from numba import njit 
import argparse as ap
from scipy.spatial.distance import pdist, squareform

METHODS = {}
def register(name):
    """Registry for methods

    Args:
        name (string): name of method.
    """    
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def maybe_timed(fn):
    """Wrapper for methods.

    Args:
        fn (function): given method
    """    
    def wrap(*args, timed=False, **kwargs):
        if not timed:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return out, dt
    return wrap


def generate_particles(scale, N=1000):
    """Generates N points randomly 
    accross given scaled space.

    Args:
        scale (float):  scaling factor for 
                        distances up to where points are spread.
        N (int, optional): Amount of points to generate. Defaults to 1000.

    Returns:
        Array: (N, 3) form of generated points.
    """
    A_x = np.random.random(N)*scale
    A_y = np.random.random(N)*scale
    A_z = np.random.random(N)*scale
    
    x = np.column_stack((A_x, A_y, A_z)) # Shape (N, 3)
    return x


@register("norm")
def distances_norm(points):
    """Uses regular norm for loop for calculating 
    distances for N points from origo.

    Args:
        points ((N, 3) array): points to calculate distances for.

    Returns:
        array: Calculated distances.
    """    
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
    """Uses reuglar norm calculation but accelerated with numba 
    for calculatint distances for N points from origo.

    Args:
        points ((N, 3) array): points to calculate distances for.

    Returns:
        array: Calculated distances.
    """    
    distance_values = np.empty(max(points.shape))
    vec_norm = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    for i in range(max(points.shape)):
        x_i, y_i, z_i = points[i]
        dist_i = vec_norm(x_i, y_i, z_i)
        distance_values[i] = dist_i
    return distance_values


@register("npnorm")
def distances_npnorm(points):
    """Uses npnorm for calculatint distances for N points from origo.

    Args:
        points ((N, 3) array): points to calculate distances for.

    Returns:
        array: Calculated distances.
    """    
    return LA.norm(points, axis=1)


@register("einsum")
def distances_einsum(points):
    """Uses einsum for calculatint distances for N points from origo.

    Args:
        points ((N, 3) array): points to calculate distances for.

    Returns:
        array: Calculated distances.
    """    
    return np.sqrt(np.einsum('ij,ij->i',points,points))


@register("pdist")
def distances_pdist(points):
    """Uses pdist for calculatint distances for N points from origo.

    Args:
        points ((N, 3) array): points to calculate distances for.

    Returns:
        array: Calculated distances.
    """    
    # Stack origin as last point
    pts_aug = np.vstack([points, np.zeros(3)])
    # Pairwise condensed distances
    cond = pdist(pts_aug)
    full = squareform(cond)
    return full[-1, :-1]


def compute(x, method="norm", timed=False, warmup=True):
    """Method caller for computation.

    Args:
        x (array): point array.
        method (str, optional): Method name. Defaults to "norm".
        timed (bool, optional): Is the computation timed. Defaults to False.

    Returns:
        function: timed function if timed, otherwise without timing.
    """    
    fn = METHODS[method]
    # JIT compile once before timing
    if timed and warmup and method == "norm-numba":
        fn(x)                      # compile on first call (not timed)
    wrapped = maybe_timed(fn)
    return wrapped(x, timed=timed)


def main(args):
    """Main function that parses parameters, computes
    given data and prints output to console.

    Args:
        args (NameSpace): CLI parameters

    Raises:
        ValueError: Given when no methods are pasted.
    """    
    selected = args.methods
    methods = list(METHODS.keys()) if "all" in selected else selected
    
    if len(selected) == 0: raise ValueError("You must give one or more methods.")
    
    N_points = args.n
    scaling_value = args.scaling
    timed = args.timed
    print_d = args.print_distance
    points = generate_particles(scaling_value, N_points)

    r_values = {}
    dt_values = {}
    
    for name in methods:
        if timed: r_values[name], dt_values[name] = compute(points, method=name, timed=timed)    
        else: r_values[name] = compute(points, method=name, timed=timed)     
    if print_d: 
        for name in r_values:
            print(f"\n Point distances for method {name}: \n" + str(r_values[name]))
        for name in dt_values:
            print(f"\n Running time for method {name}: \n" + str(dt_values[name]))
    else: 
        for name in dt_values:
            print(f"\n Running time for method {name}: \n" + str(dt_values[name]))
    if not timed:
        for name in r_values:
            print(f"\n Point distances for method {name} wihtout timing: \n" + str(r_values[name]))
    return


def build_parser():
    """Build CLI parser.

    Returns:
        NameSpace: parser for CLI commands
    """    
    parser = ap.ArgumentParser(
        description=
        "Calculates the distance from origo to point for N points. Timable wrapper available")
    
    parser.add_argument("-n", type=int, default=1000, 
                        help="Ampount of points")
    parser.add_argument("-s", "--scaling", type=float, default=5.0, 
                        help="Scaling factor for dimension in which the points are generated.")
    parser.add_argument("-t", "--timed", action="store_false", 
                        help="Time the function runtime")
    parser.add_argument("-pd", "--print_distance", action="store_false",
                        help="Print distance values calculated for each point using each defined method")
    parser.add_argument(
        "-m", "--methods",
        choices=sorted(list(METHODS.keys())+["all"]),
        nargs="*",
        default="",
        help="One or more methods to run (or 'all')"
    )
    return parser


if __name__ == '__main__':
    # DEBUG = True
    DEBUG = False
    parser = build_parser()
    if DEBUG:
        fake = ["-m", "norm", "npnorm"]
        args = parser.parse_args(fake)
    else:
        args = parser.parse_args()
    main(args)

