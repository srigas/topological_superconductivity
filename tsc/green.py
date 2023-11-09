"""
The green.py file contains all the functions necessary for the construction of the Green's functions and all relevant calculations
"""

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

