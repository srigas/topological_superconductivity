"""
The bands.py file takes an input of k-space high symmetry points, translates them into directios and then
uses the self-consistently calculated Hamiltonian inputs to make band diagrams.
"""

import numpy as np
from typing import Dict, List, Tuple

# This function returns the high-symmetry points for every type of available lattice
def get_points_dict(latt: str, ALAT: float) -> Dict[str,np.ndarray]:

    assert latt in ('SC', 'BCC', 'FCC'), "Not implemented. Choose from 'SC', 'BCC', or 'FCC'"
    points: Dict[str,np.ndarray] = {}
    points['Γ'] = np.array([0.0,0.0,0.0])
    k_const = np.pi/ALAT

    if latt == 'SC':
        points['R'] = k_const*np.array([1.0,1.0,1.0])
        points['X'] = k_const*np.array([0.0,1.0,0.0])
        points['M'] = k_const*np.array([1.0,1.0,0.0])
        # also account for foreign letters
        points['Χ'] = k_const*np.array([0.0,1.0,0.0])
        points['Μ'] = k_const*np.array([1.0,1.0,0.0])
    elif latt == 'BCC':
        points['H'] = k_const*np.array([0.0,0.0,2.0])
        points['P'] = k_const*np.array([1.0,1.0,1.0])
        points['N'] = k_const*np.array([0.0,1.0,1.0])
        # also account for foreign letters
        points['Η'] = k_const*np.array([0.0,0.0,2.0])
        points['Ρ'] = k_const*np.array([1.0,1.0,1.0])
        points['Ν'] = k_const*np.array([0.0,1.0,1.0])
    else:
        points['X'] = k_const*np.array([0.0,2.0,0.0])
        points['L'] = k_const*np.array([1.0,1.0,1.0])
        points['W'] = k_const*np.array([1.0,2.0,0.0])
        points['U'] = k_const*np.array([0.5,2.0,0.5])
        points['K'] = k_const*np.array([1.5,1.5,0.0])
        # also account for foreign letters
        points['Χ'] = k_const*np.array([0.0,2.0,0.0])
        points['Κ'] = k_const*np.array([1.5,1.5,0.0])
    
    return points

# This function takes an input string of points and translates it into k-space vectors
def get_sym_points(letters: List[str], latt: str, ALAT: float) -> np.ndarray:

    # Call the helper function to get the correct dict
    points_dict = get_points_dict(latt, ALAT)

    # Extract the letters for the high symmetry points
    assert all(char in points_dict for char in letters), "There are points which do not correspond to high-symmetry points for the given lattice."

    sym_pts = np.zeros((len(letters),3))
    # Turn letters into vectors
    for idx, letter in enumerate(letters):
        sym_pts[idx,:] = points_dict[letter]
    
    return sym_pts

# This function returns the necessary k-points for the calculations, as well as an additional array
# to be used for plotting, in order to depict the correct distances between high symmetry points
def get_band_KPTS(sym_pts: np.ndarray, k_mesh) -> Tuple[np.ndarray, np.ndarray]:
    
    # Check if mesh is given as integer, in which case we transform it to a uniform list:
    if isinstance(k_mesh,int): k_mesh = [k_mesh]*(len(sym_pts)-1)
    k_mesh = np.array(k_mesh,dtype=int)

    # Now start getting KPTS based on the sym_pts array
    band_KPTS = np.zeros((k_mesh.sum()+1,3))
    # Also get the relative magnitudes for the plot
    x_for_plot = np.zeros((k_mesh.sum()+1))
    
    cum_ct = 0.0
    for sdx in range(len(sym_pts)-1):
        # Get starting and ending indices to populate the band_KPTS array
        ini, fin = sdx*k_mesh[sdx], (sdx+1)*k_mesh[sdx]
        # Get the two vectors that define the direction
        k_dir = sym_pts[sdx+1] - sym_pts[sdx]
        
        # Get the mesh for this direction
        increments = np.linspace(0,1,k_mesh[sdx]+1)
        # reshaping is necessary for broadcasting
        kpts = sym_pts[sdx] + k_dir*(increments.reshape(-1,1))
    
        # Also calculate the magnitude of the direction, to populate the x_for_plot array
        mag = np.linalg.norm(k_dir)
        xpts = cum_ct + mag*increments
    
        # Make sure that we do not double-count the high symmetry points
        if sdx == len(sym_pts)-2:
            band_KPTS[ini:,:] = kpts
            x_for_plot[ini:] = xpts
        else:
            band_KPTS[ini:fin,:] = kpts[:-1,:]
            x_for_plot[ini:fin] = xpts[:-1]
            
        cum_ct += mag

    return band_KPTS, x_for_plot, k_mesh

# This function returns the contribution of each atom to the corresponding eigenenergy
def get_weights(N_b: int, N_k: int, band_vecs: np.ndarray) -> np.ndarray:
    # Get indices, where N_b corresponds to the number of atoms and 4 is due to the 4plity in eigenvalues
    m_indices, p_indices, k_indices = np.indices((N_b, 4, N_k))
    
    # Calculate the row indices into 'band_vecs' based on 'm', 'p', and 'k'
    row_indices = m_indices + p_indices * N_b  # shape (N_b, 4, band_N_k)
    
    # Use indexing to select the required elements from 'band_vecs'
    # Need to expand dimensions of row_indices to match the indexing of band_vecs
    expanded_row_indices = row_indices[:, :, :, np.newaxis]  # shape (N_b, 4, band_N_k, 1)
    expanded_k_indices = k_indices[:, :, :, np.newaxis]  # shape (N_b, 4, band_N_k, 1)
    column_indices = np.arange(4*N_b)  # shape (4*N_b,)

    # Now index band_vecs using the constructed indices
    selected_v = band_vecs[expanded_k_indices, expanded_row_indices, column_indices]  # shape (N_b, 4, band_N_k, 4*N_b)
    
    # Compute the absolute values, square them, and sum along the 'p' dimension
    weights = np.sum(np.abs(selected_v)**2, axis=1)  # shape (N_b, band_N_k, 4*N_b)

    return weights
