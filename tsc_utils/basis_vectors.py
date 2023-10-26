"""
The basis_vectors.py file holds functions that generate the required parameters for the problem's basis atoms.
The two main functions [1] create a single basis atom or [2] create a slab. Additional functions create
different types of configurations, which may not be as universal.
"""

import numpy as np

# Function that returns a single basis atom as a (1, 11) numpy array
def single_atom(x: float, y: float, z: float, 
                atom_type: int, E_0: float, U: float, n_0: float, 
                B_x: float, B_y: float, B_z: float, V: float) -> np.ndarray:
    
    atom = np.array([[x, y, z, atom_type, E_0, U, n_0, B_x, B_y, B_z, V]])
    return atom

# Function that returns a slab along a specified axis, with given starting index and length
def slab(start_index: int, slab_length: int, fixed_coord_1: float, fixed_coord_2: float, axis: str,
         atom_type: int, E_0: float, U: float, n_0: float, B_x: float, B_y: float, B_z: float, V: float) -> np.ndarray:
    
    assert axis in ('x', 'y', 'z'), "Invalid axis. Choose from 'x', 'y', or 'z'"
    
    # Infer end_index
    end_index = start_index + slab_length - 1
    
    # Create a range of indices for the varying coordinate
    varying_coord = np.arange(float(start_index), float(end_index) + 1.0).reshape(-1, 1)
    
    # Create arrays for the fixed coordinates
    fixed_coords_1 = np.full((slab_length, 1), fixed_coord_1)
    fixed_coords_2 = np.full((slab_length, 1), fixed_coord_2)
    
    # Concatenate the coordinates based on the axis
    if axis == 'x':
        coords = np.hstack((varying_coord, fixed_coords_1, fixed_coords_2))
    elif axis == 'y':
        coords = np.hstack((fixed_coords_1, varying_coord, fixed_coords_2))
    else:
        coords = np.hstack((fixed_coords_1, fixed_coords_2, varying_coord))
    
    # Create arrays for the other atom parameters
    other_params = np.array([atom_type, E_0, U, n_0, B_x, B_y, B_z, V])
    other_params_repeated = np.tile(other_params, (slab_length, 1))
    
    # Concatenate the coordinates and other parameters to form the slab
    slab = np.hstack((coords, other_params_repeated))
    
    return slab

