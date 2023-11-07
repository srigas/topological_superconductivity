"""
The utilities.py file holds functions that are relevant for many different applications, either for setup
(e.g. getting Bravais vectors) or for defining useful functions (e.g. the Fermi function)
"""

import numpy as np
from typing import Tuple

# Vectorized Fermi function
def fermi(E: np.ndarray, T: float, k_B: float) -> np.ndarray:
    if T < 1e-8:
        result = np.where(E > 0, 1.0, np.where(E == 0, 0.5, 0.0))
    else:
        term = E / (k_B * T)
        # Clip term values to avoid overflow/underflow
        clipped_term = np.clip(term, -500.0, 500.0)
        result = 1.0 / (np.exp(clipped_term) + 1.0)
    return result

# Function that returns a (2NCELLS+1)x(2NCELLS+1)x(2NCELLS+1) grid
def get_RPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, NCELLS: int) -> np.ndarray:
    
    rng = np.arange(-NCELLS, NCELLS + 1)
    i, j, k = np.meshgrid(rng, rng, rng, indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    RPTS = (i[:, np.newaxis] * a_1 + j[:, np.newaxis] * a_2 + k[:, np.newaxis] * a_3)
    
    return RPTS

# Function that returns the k-mesh
def get_KPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, N_x: int, N_y: int, N_z: int) -> np.ndarray:
    
    # Calculate the volume of the unit cell
    volume = np.dot(a_1, np.cross(a_2, a_3))
    
    # Calculate the reciprocal space vectors
    b_1 = 2.0 * np.pi * np.cross(a_2, a_3) / volume
    b_2 = 2.0 * np.pi * np.cross(a_3, a_1) / volume
    b_3 = 2.0 * np.pi * np.cross(a_1, a_2) / volume
    
    # Calculate the total number of different wavevectors in reciprocal space
    Ntot = N_x * N_y * N_z
    
    # Initialize KPTS
    KPTS = np.zeros((Ntot, 3))
    
    # Generate ranges for c_1, c_2, c_3
    c_1_range = np.arange(1, N_x + 1) / N_x
    c_2_range = np.arange(1, N_y + 1) / N_y
    c_3_range = np.arange(1, N_z + 1) / N_z
    
    # Create a grid of c_1, c_2, c_3 values
    c_1, c_2, c_3 = np.meshgrid(c_1_range, c_2_range, c_3_range, indexing='ij')
    
    # Reshape the grid arrays to 1D arrays
    c_1, c_2, c_3 = c_1.ravel(), c_2.ravel(), c_3.ravel()
    
    # Calculate KPTS using broadcasting
    KPTS[:, 0] = b_1[0] * c_1 + b_2[0] * c_2 + b_3[0] * c_3
    KPTS[:, 1] = b_1[1] * c_1 + b_2[1] * c_2 + b_3[1] * c_3
    KPTS[:, 2] = b_1[2] * c_1 + b_2[2] * c_2 + b_3[2] * c_3
    
    return KPTS

# Function that counts the number of neighbours for every atom, given R_max
def get_nndists(RPTS: np.ndarray, TPTS: np.ndarray, R_max: float) -> np.ndarray:

    N_b : int = TPTS.shape[0]
    # Array to hold the number of neighbours for every atom
    num_neighbs = np.zeros((N_b), dtype=int)

    for i in range(N_b):
        for j in range(N_b):
            ttprime = TPTS[j] - TPTS[i]
            for rpoint in RPTS:
                dist = np.linalg.norm(rpoint + ttprime)
                if (dist < R_max + 1e-5) and (dist > 1e-5):
                    num_neighbs[i] += 1
                    
    return num_neighbs

# Get the type and distance connection vectors
def get_connections(max_neighb: int, RPTS: np.ndarray, TPTS: np.ndarray, R_max: float) -> Tuple[np.ndarray, np.ndarray]:

    N_b: int = TPTS.shape[0]
    # Arrays to hold the required results
    type_IJ = np.zeros((N_b, max_neighb), dtype=int)
    Rvec_ij = np.zeros((N_b, max_neighb, 3))

    for i in range(N_b):
        added = 0
        for j in range(N_b):
            ttprime = TPTS[j] - TPTS[i]
            for rpoint in RPTS:
                vec = rpoint + ttprime
                dist = np.linalg.norm(vec)
                if (dist < R_max + 1e-5) and (dist > 1e-5):
                    # Get the type of atom j that is a neighbour of atom i
                    # excluding self-interactions
                    type_IJ[i, added] = j
                    # Get the vector connecting atom j to atom i
                    Rvec_ij[i, added, :] = vec
                    added += 1

    return type_IJ, Rvec_ij

# Simple function that returns an energy mesh to perform DoS calculations
def get_dos_mesh(E_min: float, E_max: float, intervals: int) -> np.ndarray:

    # Get the energies and make an energy grid
    Es = np.linspace(E_min,E_max,intervals)

    return Es

# Function to get DoS using the energy eigenvalues and the electron eigenvectors as weights
# Surprisingly, vectorizing this in terms of energies results into a slower execution time
def get_dos(E_vecs: np.ndarray, E_vals: np.ndarray, Es: np.ndarray, N_b: int, Γ: float, N_k: int) -> np.ndarray:

    intervals = Es.shape[0]
    
    # Setup an array to hold the DoS values
    DOS = np.zeros((N_b,intervals))

    # Extract wavefunctions for electrons
    u_ups, u_downs = E_vecs[:N_b, :], E_vecs[N_b:2*N_b, :]
    u_up_squared = np.abs(u_ups)**2
    u_down_squared = np.abs(u_downs)**2
    
    for idx, E in enumerate(Es):
        DOS[:,idx] = (Γ/np.pi)*(1.0/N_k)*((u_up_squared+u_down_squared)/(Γ**2 + (E_vals-E)**2)).sum(axis=1)

    return DOS

# Function that creates an energy mesh, including integration weights, to be used for
# integrals that are required to get observables
def get_int_mesh() -> :

    
    
    return