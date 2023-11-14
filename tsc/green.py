"""
The green.py file contains all the functions necessary for the construction of the Green's functions and all relevant calculations
"""

import numpy as np

# This function calculates memory requirements for the fully vectorized operations
# involved in the calculation of the Green's Function
def check_mem(N_k: int, N_b: int, N_at: int, N_imp: int, N_E: int) -> None:

    # Convert results from bits into GBs
    factor = 1.0/(8*1024*1024*1024)
    
    # GFK per Energy, at some point requires a (N_k, 4N_at, 4N_at, 4N_b) matrix
    gfk_1 = factor*128*N_k*4*N_at*4*N_at*4*N_b
    print(f"The memory requirements for the k-space Green's Function per energy: {gfk_1:.4f} GigaBytes")
    # GFK total, at some point requires a (N_k, 4N_at, 4N_at, 4N_b, N_E) matrix
    gfk_2 = gfk_1*N_E
    print(f"The memory requirements for the total k-space Green's Function is: {gfk_2:.4f} GigaBytes")
    # GFR per Energy, at some point requires a (N_k, 4*N_imp, 4*N_imp) matrix
    gfr_1 = factor*128*N_k*4*N_imp*4*N_imp
    print(f"The memory requirements for the real space Green's Function per energy: {gfr_1:.4f} GigaBytes")
    # GFR total, at some point requires a (N_k, 4*N_imp, 4*N_imp, N_E) matrix
    gfr_2 = gfr_1*N_E
    print(f"The memory requirements for the total real space Green's Function is: {gfr_2:.4f} GigaBytes")
    
    return

# This function returns the fourier exponentials for the inverse Fourier transform
def green_fourier(IPTS: np.ndarray, KPTS: np.ndarray) -> np.ndarray:

    # Compute Rvec for all i, j combinations, by subtracting j-i
    Rvecs = IPTS[:, np.newaxis, :] - IPTS[np.newaxis, :, :]
    # Rvecs is now j,i,3 while KPTS is k,3
    # Perform dot product using Einstein notation
    dot_products = np.einsum('km,jim->kij', KPTS, Rvecs)
    # Calculate exponentials
    expons = np.exp(-1j * dot_products)

    return expons

# This function returns the final real-space Green's Function for the host system
# using fully vectorized operations, if the system can handle them
def get_green_vec(Es: np.ndarray, E_vecs: np.ndarray, E_vals: np.ndarray, indices: np.ndarray, fourier: np.ndarray) -> np.ndarray:

    # Get N_k
    N_k = E_vals.shape[0]
    
    # Get the arrays required for N_at, N_imp
    unq_idxs, at_indices = np.unique(indices, return_inverse=True)
    N_at = unq_idxs.shape[0]
    N_imp = at_indices.shape[0]
    
    # Setup the energies, which are N_E distinct values, for broadcasting
    Es_exp = Es[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :] # Shape (1, 1, 1, 1, N_E)
    
    # Given the unique atom indices, we extend them so that we get all 4 wavefunctions for each atom,
    # using the alternate description of E_vecs
    unq_idxs_ext = (4*unq_idxs[:, None] + np.arange(4, dtype=int)).flatten()
    
    # Get the constituents and shape them properly for broadcasting
    Phi_i = E_vecs[:, unq_idxs_ext, np.newaxis, :, np.newaxis] # Shape (N_k, 4*N_at, 1, 4*N_b, 1)
    Phi_j = np.conj(E_vecs[:, np.newaxis, unq_idxs_ext, :, np.newaxis]) # Shape (N_k, 1, 4*N_at, 4*N_b, 1)
    E_vals_exp = E_vals[:, np.newaxis, np.newaxis, :, np.newaxis] # Shape: (N_k, 1, 1, 4*N_b, 1)
    
    # Calculate GFK by summing over the 4*N_b-dimensional axis
    GFK = ((Phi_i*Phi_j)/(Es_exp-E_vals_exp)).sum(axis=3)
    
    # Make mapping between all indices, including duplicate atom indices, as these are now impurity indices
    # and not basis atom indices
    row_indices = (4*at_indices[:, None] + np.arange(4, dtype=int)).flatten()
    
    # Extend fourier depending on the at_indices
    fg_ext = fourier[:, at_indices[:, None], at_indices]
    # Make each element a 4x4 matrix
    fg_large = np.repeat(np.repeat(fg_ext, 4, axis=2), 4, axis=1)
    # Reshape fourier
    fg_exp = fg_large[:, :, :, np.newaxis]
    
    # Calculate GFR
    GFR = (1.0/N_k)*(fg_exp*GFK[:, row_indices[:, None], row_indices, :]).sum(axis=0) # Shape (4*N_imp, 4*N_imp, N_E)

    return GFR
