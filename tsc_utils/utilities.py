"""
The utilities.py file holds functions that are relevant for many different applications, either for setup
(e.g. getting Bravais vectors) or for defining useful functions (e.g. the Fermi function)
"""

import numpy as np
from typing import Tuple

# Function that returns the Bravais vectors of the corresponding lattice as a tuple
def get_bravais(vec_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vec_type == "SC":
        a_1 = np.array([1.0,0.0,0.0])
        a_2 = np.array([0.0,1.0,0.0])
        a_3 = np.array([0.0,0.0,1.0])
    elif vec_type == "BCC":
        a_1 = np.array([0.5,0.5,-0.5])
        a_2 = np.array([-0.5,0.5,0.5])
        a_3 = np.array([0.5,-0.5,0.5])
    elif vec_type == "FCC":
        a_1 = np.array([0.5,0.5,0.0])
        a_2 = np.array([0.0,0.5,0.5])
        a_3 = np.array([0.5,0.0,0.5])
    else:
        print("Not implemented")
    return a_1, a_2, a_3

# Function that returns the 2D Pauli matrices as a tuple
def get_pauli() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_0 = np.array([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=complex)
    s_1 = np.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=complex)
    s_2 = np.array([[0+0j, 0-1j], [0+1j, 0+0j]], dtype=complex)
    s_3 = np.array([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=complex)
    
    return s_0, s_1, s_2, s_3

# Fermi function
def Fermi(E: float, T: float, KB: float) -> float:
    if T < 1e-8: # Check for zero temperature
        if E < 0.0:
            return 1.0
        elif E == 0.0:
            return 0.5
        else:
            return 0.0
    else:
        term = E / (KB * T)
        # to avoid overflow errors
        if term > 500.0:
            return 0.0
        elif term < -500.0:
            return 1.0
        else:
            return 1.0 / (np.exp(term) + 1.0)

# Function that returns a (2NCELLS+1)x(2NCELLS+1)x(2NCELLS+1) grid
def get_RPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, NCELLS: int) -> np.ndarray:
    rng = np.arange(-NCELLS, NCELLS + 1)
    i, j, k = np.meshgrid(rng, rng, rng, indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    RLATT = (i[:, np.newaxis] * a_1 + j[:, np.newaxis] * a_2 + k[:, np.newaxis] * a_3).T
    return RLATT

# Function that returns the k-mesh
def get_KPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, N_x: int, N_y: int, N_z: int) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    # Calculate the volume of the unit cell
    volume = np.dot(a_1, np.cross(a_2, a_3))
    
    # Calculate the reciprocal space vectors
    b_1 = 2.0 * np.pi * np.cross(a_2, a_3) / volume
    b_2 = 2.0 * np.pi * np.cross(a_3, a_1) / volume
    b_3 = 2.0 * np.pi * np.cross(a_1, a_2) / volume
    
    # Calculate the total number of different wavevectors in reciprocal space
    Ntot = N_x * N_y * N_z
    
    # Initialize KPTS
    KPTS = np.zeros((3, Ntot))
    
    # Generate ranges for c_1, c_2, c_3
    c_1_range = np.arange(1, N_x + 1) / N_x
    c_2_range = np.arange(1, N_y + 1) / N_y
    c_3_range = np.arange(1, N_z + 1) / N_z
    
    # Create a grid of c_1, c_2, c_3 values
    c_1, c_2, c_3 = np.meshgrid(c_1_range, c_2_range, c_3_range, indexing='ij')
    
    # Reshape the grid arrays to 1D arrays
    c_1, c_2, c_3 = c_1.ravel(), c_2.ravel(), c_3.ravel()
    
    # Calculate KPTS using broadcasting
    KPTS[0, :] = b_1[0] * c_1 + b_2[0] * c_2 + b_3[0] * c_3
    KPTS[1, :] = b_1[1] * c_1 + b_2[1] * c_2 + b_3[1] * c_3
    KPTS[2, :] = b_1[2] * c_1 + b_2[2] * c_2 + b_3[2] * c_3
    
    return KPTS