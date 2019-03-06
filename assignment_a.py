# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:34:18 2019

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import scipy as sc
from functools import partial
import numpy.random as random

def legendre_prime(x, n):
    """Calculate the first derivative of the nth Legendre Polynomial recursively.

    Args:
        x (float, np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = n * legendre(n - 1)(x) - n * x * legendre(n)(x)
    return legendre_p


def legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp


def poly(x, n):
    """Define the polynomial function p(x)
    Args:
        x (float/np.array) : domain
        n (int) : degree of the polynomial
    
    Returns:
        values (float/np.array) : polynomial evaluated in the domain
    """
    p = (1-x*x)*legendre_prime(x,n)
    return p 

def poly_prime(x, n):
    """Define the derivative of the polynomial function p(x)
    Args:
        x (float/np.array) : domain
        n (int) : degree of the polynomial
    
    Returns:
        values (float/np.array) : polynomial evaluated in the domain
    """
    pp = (1-x*x)*legendre_double_prime(x,n) - 2*x*legendre_prime(x,n)
    return pp

#==============================================================================
# 
# xtab = np.arange(-1,1.01,0.01)
# y1tab = poly_prime(xtab, 5)
# y2tab = poly(xtab,5)
# 
# plt.plot(xtab,y1tab)
# plt.plot(xtab,y2tab)
# plt.axhline(y=0, color= 'k')
# plt.show()
# 
#==============================================================================

def newton_method(f, dfdx, x_0, iter_max=100, min_error=1e-14):
    """Newton method for rootfinding.

    Args:
        f (function object): function to find root of, f: float -> float
        dfdx (function object): derivative of f, dfdx: float -> float
        x_0 (float): initial guess of root
        iter_max (int): max number of iterations
        min_error (float): min allowed error

    Returns:
        x_tilde (float): approximation of root of f
        x_history (np.array): history of convergence, [x_0, x_1, ..., x_N]
    """
   
    i = 0
    x = x_0
    x_history = [x]
    error = abs(f(x))
    while (i < iter_max or error > min_error):
        x = x - f(x)/dfdx(x)
        x_history.append(x)
        error = abs(f(x))
        i+= 1
    
    x_tilde = x
    x_history = np.array(x_history)
    return x_tilde, x_history


def chebychev(n):
    """Calculate roots of the n+1 Chebychev polynomial of the first kind.
    
    Returns:
        nodal_pts (np.array): n + 1 Chebychev grid points
    """
    i = np.arange(0, n+1)
    # nodal points in [-1,1]
    return np.cos((2.*i + 1.) / (2. * (n+1.)) * np.pi)
        
###############################################################        
def gauss_lobatto(n):
    """Calculate the roots of (1+x**2)*L'_n(x) and therefore find the n + 1 points of the LGL grid.
    Args:
        n : # subdivision of the grid -> correspond to n+1 points
    Returns:
        grid_points (np.array): n + 1 grid points
    """
    # Chebychev nodes as initial guesses for the Newton method.
#    x_0 = chebychev(5)
    x_0 = np.cos(np.arange(1., n) / n * np.pi)

    
    # We fix the degree of the polynomial, pass these functions to the Newton Method.
    poly_fn = partial(poly, n=n)
    poly_prime_fn = partial(poly_prime, n=n)
    
    grid_points = np.empty(n + 1)
    # Last and first pts are fixed for every n
    grid_points[-1] = -1
    grid_points[0] = 1
    
    # Newton's method to find the root. Do to not overwrite the grid_points[0], grid_points[-1], i.e.
    # Use x_0[i] to find the grid_points[i+1]
    # TODO ... find the grid points using Newton's method for each point
    
    for i in range(1,n):
        grid_pt, history = newton_method(poly_fn, poly_prime_fn, x_0[i-1])
        grid_points[i] = grid_pt
    
    return grid_points


#==============================================================================
# def chebychev(n):
#     """Calculate roots of the n+1 Chebychev polynomial of the first kind.
#     
#     Returns:
#         nodal_pts (np.array): n + 1 Chebychev grid points
#     """
#     i = np.arange(0, n+1)
#     # nodal points in [-1,1]
#     return np.cos((2.*i + 1.) / (2. * (n+1.)) * np.pi)
# 
#==============================================================================
def plot_grids(grids, labels):
    """Plot multiple grids.

    Args:
        grids (list): list of grids
        labels (list): labels for each grid (strings)
    """
    for i, grid in enumerate(grids):
        plt.plot(grid, np.ones(np.size(grid)) * i, '-o', label=labels[i])
    plt.xlabel(r'$\xi$')
    plt.ylim(-1, np.ndim(grids) + 1)
    plt.legend(), plt.show()
        
# plot of the grids
n = 5
grid_lgl = gauss_lobatto(n)
grid_ch = chebychev(n)
#plot_grids([grid_lgl, grid_ch], ['lobatto grid', 'chebychev grid'])

def hash_string2int(net_id, maxint):
    """Convert string to integer in the range [0, maxint)."""
    sum_chars = 0
    for char in net_id:
        sum_chars += ord(char)
    return sum_chars % maxint

def rand(seed, n):
    """Custom pseudo-random numbers approximately uniform on [0,1]. Needed for
    consistency across Python versions/machines etc. where numpy implementation
    may change. (We're generating problem-inputs from usernames.)

    Args:
      seed (integer): Random number seed, required on every call to this fn.
      n (integer): Number of random numbers to return.

    Return:
      out (list of floats): Random numbers on [0,1]
    """
    out = []
    m, a, c = 2**32, 1103515245, 12345
    for i in range(n):
        seed = (a * seed + c) % m
        out.append(float(seed) / (m-1))
    return out


def get_data(x, net_id):
    """Test function unique for each student based on netid"""
    seed = hash_string2int(net_id, 2**32)
    a = rand(seed, 5)
    return 0.1*a[4]*np.cos(10.*x) + a[3]*np.cos(3.*x) + a[2]*np.sin(2. * x) + a[1]*np.sin(0.5 * x) + a[0]

net_id = 'acummins'

def scale_grid(grid, a, b):
    """Linear scaling of the grid from [-1, 1] to [a, b].
    
    Args:
        grid (np.array): grid data points
        a, b (float): start, end interval        
    Return:
        scaled_grid (np.array): grid scaled to interval [a,b]
    """
    scaled_grid = (b - a)/2. * (grid + 1.)
    return scaled_grid[::-1]

N = 8
grid = gauss_lobatto(N)          # LGL grid on [0,1]
grid = scale_grid(grid, 0, 2)    # LGL grid on [0,2]

# Get the data (you can use this function to get your data on any grid)
data = get_data(grid, net_id)

def basis_lagrange(x, grid):
    """
    Args:
        x (float): Location at which to evaluate the basis.
        grid (np.array): Grid nodes for which to construct basis.
    Return:
        phi (np.array): Basis functions at x, one per grid-point.
    """
#==============================================================================
#     from functools import reduce, operator 
#     p = [(x - grid[j])/(grid[i] - grid[j]) for i in range((len(grid))) if j != i]
#     return reduce(operator.mul, p)
# 
# 
#==============================================================================

    n = len(grid)
    l = np.zeros([n])
    idx = list(range(0,n))
    
    for i in idx:
        idx = list(range(0,n))
        basislst = []
        jj = idx
        jj.remove(i)
        for j in jj:
            basislst.append((x-grid[j])/(grid[i]-grid[j])) 
        
        pbasis= np.array(basislst)
        basis = np.prod(pbasis)
              
        l[i] = basis           



    return l

def reconstruct(x, grid, data, basis):
    """Reconstruct the interpolating function
    Args:
        x (np.array): array of locations to evaluate interpolant
        grid (np.array): node locations 
        data (np.array): data values at each node
        basis (function object) : your function `basis_lagrange()`
    
    Returns:
        interpolant (np.array) : interpolant function evaluated in a domain of choice
    """
    interpolant = np.zeros(len(x))
    for i in range(len(x)) :
        interpolant[i] = basis(x[i],grid).T @ data

    return interpolant

x_out = np.linspace(0,2,101)
interpolant_re = reconstruct(x_out, grid, data, basis_lagrange)
# Plot original function, data and interpolant

plt.plot(x_out, get_data(x_out, net_id))
plt.plot(grid, data, 'o')
plt.plot(x_out, interpolant_re)

value = reconstruct(np.array([1.25]),grid,data,basis_lagrange)
print(value)








