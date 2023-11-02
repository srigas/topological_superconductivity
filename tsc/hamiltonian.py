"""
The hamiltonian.py file contains all the functions necessary for the construction of the Hamiltonian matrix, to be diagonalized
as the solution of the BdG Equations. There are two types of hamiltonians: the metal one and the superconducting one.
"""

import numpy as np
import numba

import copy

# This function calculates the hopping element constants, t_0
# It is far from optimized, however in this case readability is more important than efficiency,
# because it is a function that runs once, not through every self-consistency cycle
def hopping_consts(hop_mat: np.ndarray, atom_types: np.ndarray, TPTS: np.ndarray, RPTS: np.ndarray, R_0: float) -> np.ndarray:
    N_unique: int = hop_mat.shape[0]
    N_b: int = atom_types.shape[0]
    # Array to hold the hopping element constants
    t_0 = np.zeros((N_unique,N_unique))
    # Array to hold the minimum distances between atom types
    # Initialize it to a relatively large value
    NN_IJ = 1000*np.ones((N_unique,N_unique))
    
    # Two loops over distinct atom types - note that indexing in python starts from 0 and
    # this is also where the atom type indices start from, so this works
    for Itype in range(N_unique):
        for Jtype in range(N_unique):
            
            # Two loops over all atoms
            for i in range(N_b):
                for j in range(N_b):
                    # Check if we have an I-J agreement
                    if (atom_types[i] == Itype) and (atom_types[j] == Jtype):
                        # If so, calculate the distance between the basis atoms
                        ttprime = TPTS[j] - TPTS[i]
                        # Loop over all lattice points - this is necessary in case the basis atoms' coordinates are not given
                        # in such a way that they are as close as possible
                        for rpoint in RPTS:
                            dist = np.linalg.norm(rpoint + ttprime)
                            # check if the distance is the lowest one yet as long as it is not 0
                            if (NN_IJ[Itype,Jtype] > dist) and (dist > 1e-5):
                                NN_IJ[Itype,Jtype] = dist
            # Get the exponential term
            expon = np.exp(NN_IJ[Itype,Jtype]/R_0)
            # Finally, get the t_0 element
            t_0[Itype,Jtype] = hop_mat[Itype,Jtype]*expon

    return t_0

# This function calculates the full hopping elements
def hopping_elements(type_IJ: np.ndarray, num_neighbs: np.ndarray, Rvec_ij: np.ndarray, atom_types: np.ndarray, t_0: np.ndarray, R_0: float) -> np.ndarray:

    N_b, max_neighb = type_IJ.shape
    t = np.zeros((N_b,max_neighb))
    
    for i in range(N_b):
        for j in range(num_neighbs[i]):
            # Get the connection vector between the two atoms
            rpoint = Rvec_ij[i,j,:]
            # Get the type of atom for the i atom
            Iatom = atom_types[i]
            # Get the j-th atom index
            Jindex = type_IJ[i,j]
            # Get the type of atom for the j atom
            Jatom = atom_types[Jindex]
            # Get the corresponding hopping constant
            const = t_0[Iatom,Jatom]
            # Finally, the hopping element
            t[i,j] = -const*np.exp(-np.linalg.norm(rpoint)/R_0)
        
    return t

# This function returns the fourier exponentials
def get_exponentials(Rvec_ij: np.ndarray, KPTS: np.ndarray) -> np.ndarray:

    # Expand KPTS and Rvec_ij
    KPTS_expanded = KPTS[:, :, None, None]  # Shape: (N_k, 3, 1, 1)
    Rvec_ij_expanded = Rvec_ij[None, :, :, :]  # Shape: (1, N_b, max_neighb, 3)

    # Transpose KPTS_expanded to match the last dimension with Rvec_ij_expanded
    KPTS_transposed = np.transpose(KPTS_expanded, (0, 2, 3, 1))  # Shape: (N_k, 1, 1, 3)

    # Compute the dot product
    dot_product = np.sum(KPTS_transposed * Rvec_ij_expanded, axis=-1)  # Shape: (N_k, N_b, max_neighb)
    
    # Compute the exponential term
    exponentials = np.exp(-1j * dot_product) # Shape: (N_k, N_b, max_neighb)

    return exponentials

# ------------------------------------
#            HAMILTONIANS
# ------------------------------------

# This function prepares the Hamiltonian's base (i.e. everything apart from the hopping and Fourier)
# Normal Metal Version
@numba.njit
def prep_N_hamiltonian(E_0: np.ndarray, μ: float, U: np.ndarray, n: np.ndarray, n_bar: np.ndarray, B: np.ndarray, 
                       s_0: np.ndarray, s_1: np.ndarray, s_2: np.ndarray, s_3: np.ndarray) -> np.ndarray:

    N_b: int = E_0.shape[0]
    h = np.zeros((2,2), dtype=np.complex128)
    H_prep = np.zeros((2*N_b, 2*N_b), dtype=np.complex128)

    for i in range(N_b):
        h = (E_0[i] - μ + U[i]*(n[i]-n_bar[i]))*s_0 - B[i][0]*s_1 - B[i][1]*s_2 - B[i][2]*s_3

        H_prep[i, i] = h[0,0]
        H_prep[i, i+N_b] = h[0,1]
        H_prep[i+N_b, i] = h[1,0]
        H_prep[i+N_b, i+N_b] = h[1,1]

    return H_prep

# This function creates the full k-Hamiltonian, including the hopping elements
# Normal Metal Version
@numba.njit
def get_N_hamiltonian(k: int, H_prep: np.ndarray, type_IJ: np.ndarray, num_neighbs: np.ndarray, 
                      fourier: np.ndarray, t: np.ndarray) -> np.ndarray:

    H = H_prep
    N_b: int = type_IJ.shape[0]

    # loop over all atoms
    for i in range(N_b):
        # loop over all neighbours
        for j in range(num_neighbs[i]):
            # Get j-th atom index
            Jatom = type_IJ[i,j]
            # calculate Fourier constant * hopping element
            term = t[i,j]*fourier[k,i,j]

            # Append accordingly
            H[i,Jatom] += term
            H[i + N_b, Jatom + N_b] += term

    return H

# This function prepares the Hamiltonian's base (i.e. everything apart from the hopping and Fourier)
# Superconductivity Version
@numba.njit
def prep_SC_hamiltonian(E_0: np.ndarray, μ: float, U: np.ndarray, n: np.ndarray, n_bar: np.ndarray, B: np.ndarray, 
                     Δ: np.ndarray, s_0: np.ndarray, s_1: np.ndarray, s_2: np.ndarray, s_3: np.ndarray) -> np.ndarray:

    N_b: int = E_0.shape[0]
    h = np.zeros((2,2), dtype=np.complex128)
    H_prep = np.zeros((4*N_b, 4*N_b), dtype=np.complex128)

    for i in range(N_b):
        h = (E_0[i] - μ + U[i]*(n[i]-n_bar[i]))*s_0 - B[i][0]*s_1 - B[i][1]*s_2 - B[i][2]*s_3

        H_prep[i, i] = h[0,0]
        H_prep[i, i+N_b] = h[0,1]
        #H_prep[i, i+2*N_b] = 0.0 + 0.0j
        H_prep[i, i+3*N_b] = Δ[i]

        H_prep[i+N_b, i] = h[1,0]
        H_prep[i+N_b, i+N_b] = h[1,1]
        H_prep[i+N_b, i+2*N_b] = Δ[i]
        #H_prep[i+N_b, i+3*N_b] = 0.0 + 0.0j

        #H_prep[i+2*N_b, i] = 0.0 + 0.0j
        H_prep[i+2*N_b, i+N_b] = np.conj(Δ[i])
        H_prep[i+2*N_b, i+2*N_b] = -np.conj(h[0,0])
        H_prep[i+2*N_b, i+3*N_b] = np.conj(h[0,1])

        H_prep[i+3*N_b, i] = np.conj(Δ[i])
        #H_prep[i+3*N_b, i+N_b] = 0.0 + 0.0j
        H_prep[i+3*N_b, i+2*N_b] = np.conj(h[1,0])
        H_prep[i+3*N_b, i+3*N_b] = -np.conj(h[1,1])

    return H_prep

# This function creates the full k-Hamiltonian, including the hopping elements
# Superconductivity Version
@numba.njit
def get_SC_hamiltonian(k: int, H_prep: np.ndarray, type_IJ: np.ndarray, num_neighbs: np.ndarray, 
                    fourier: np.ndarray, t: np.ndarray) -> np.ndarray:

    H = H_prep
    N_b: int = type_IJ.shape[0]

    # loop over all atoms
    for i in range(N_b):
        # loop over all neighbours
        for j in range(num_neighbs[i]):
            # Get j-th atom index
            Jatom = type_IJ[i,j]
            # calculate Fourier constant * hopping element
            term = t[i,j]*fourier[k,i,j]

            # Append accordingly
            H[i,Jatom] += term
            H[i + N_b, Jatom + N_b] += term
            H[i + 2*N_b, Jatom + 2*N_b] -= term
            H[i + 3*N_b, Jatom + 3*N_b] -= term

    return H
