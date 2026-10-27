import numpy as np
from scipy.spatial import distance
from typing import Callable 

"""
This script defines v_{i}(t) as defined in the paper [citation needed]
"""
def pairwise_distances(x: np.ndarray):

    """
    Computes the pairwise distance between x_{j} and x_{i} using a Euclidean norm
    -----------------------------------------------------------------------------
    Inputs:
        x (np.ndarray): An array of position co-ordinates for all agents
    Returns:
        np.ndarray: The pairwise distances between all agents as a square matrix
    """

    return distance.squareform(distance.pdist(x, metric="euclidean"))

def generate_A(x: np.ndarray, psi: Callable[[np.ndarray], np.ndarray], A_init: np.ndarray = None):

    """
    Following the cited paper, we generate the matrix A_{x} as part of the process 
    to generate the Lapclacian
    --------------------------------------------------------------------------------
    Inputs:
        X (np.ndarray) : An array of position co-ordinats for agent 'i'
        psi (Callable): A non-negative interaction function decreasing with distance
        A_init (np.ndarray): Optional input - allows us to define a pre-defined, simple adjacency
                             matrix, which is then multiplied by pairwise distances to 
                             produce the relevant adjaceny matrix for the ODE.
                                       
    Returns:
        A (np.ndarray): The matrix A_{x}, with zeros on the diagonal, and pairwise
                        distances in the positions where i =/= j
    """

    distance_matrix = pairwise_distances(x)
    A = psi(distance_matrix)

    # If A_init defines a graph structure different from the assumed weighted, complete, undirected graph, then
    # extract out the relevant pairwise distances by multiplying A by A_init
    if (A_init is not None): return (A * A_init)

    # Otherwise, return the assumed graph structure
    np.fill_diagonal(A, 0)
    return A
    
    

def generate_laplacian(A: np.ndarray):

    """
    Returns the Laplacian, defined as L = D - A, where D denotes the square matrix
    containing the sum of interaction weights in the diagonal, and 0 otherwise
    -----------------------------------------------------------------------------------
    Inputs:
        A (np.ndarray): The matrix A_{x}, with zeros on the diagonal, and pairwise
                        distances in the positions where i =/= j
    Returns:
        L (np.ndarray): The Laplacian as defined above
    """
    return (np.diag(np.sum(A, axis=1)) - A)

def equations(x: np.ndarray, v: np.ndarray, psi: Callable[[np.ndarray], np.ndarray], A_init: np.ndarray = None):

    """
    Returns the relevant equations dx and dv in matrix format, utilizing previously 
    defined functions
    -------------------------------------------------------------------------------------
    Inputs:
        x (np.ndarray): An ndarray with initial position vectors for all agents
        v (np.ndarray): An ndarray with initial velocity vectors for all agents
        psi (Callable): A non-negative interaction function decreasing with distance
                                        
    """

    N = x.shape[0]
    A = generate_A(x, psi, A_init=A_init)
    L = generate_laplacian(A)

    dx = v
    dv = -(L @ v) / N

    return dx, dv

if __name__ == "__main__":

    # Initial test values - I randomly chose these
    v0 = np.array(([1,0,0], [0,1,0], [2,3,0]))
    x0 = np.array(([0,0,0], [0,1,0], [1,0,0]))
    psi = lambda r: 1/(1+r**2)
    A_init = np.array(([0,1,0], [1,0,1], [0,1,0]))

    dx, dv = equations(x0, v0, psi, A_init=A_init)
    print(dv)
    


