"""
The basis_atoms.py file holds functions that generate the required parameters for the problem's basis atoms.
The two main functions [1] create a single basis atom or [2] create a slab. Additional functions create
different types of configurations, which may not be as universal.
The last function, get_atom_vectors(), extracts the necessary values given a basis atom array.
"""

import numpy as np
from typing import Tuple

# Function that returns a single basis atom as a (1, 11) numpy array
def single_atom(n_1: float = 0.0, n_2: float = 0.0, n_3: float = 0.0, 
                atom_type: int = 1, E_0: float = 0.0, U: float = 0.0, n_bar: float = 1.0, 
                B_x: float = 0.0, B_y: float = 0.0, B_z: float = 0.0, Λ: float = 0.5) -> np.ndarray:
    
    atom = np.array([[n_1, n_2, n_3, atom_type, E_0, U, n_bar, B_x, B_y, B_z, Λ]])
    return atom

# Function that returns a slab along a specified axis, with given starting index and length
# Notice that here fixed_coord_1 and fixed_coord_2 correspond to cartesian and not reduced coordinates
def slab(start_index: int = 0, slab_length: int = 50, fixed_coord_1: float = 0.0, fixed_coord_2: float = 0.0, axis: str = 'z', 
         atom_type: int = 1, E_0: float = 0.0, U: float = 0.0, n_bar: float = 0.4, 
         B_x: float = 0.0, B_y: float = 0.0, B_z: float = 0.0, Λ: float = 0.5) -> np.ndarray:
    
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
    other_params = np.array([atom_type, E_0, U, n_bar, B_x, B_y, B_z, Λ])
    other_params_repeated = np.tile(other_params, (slab_length, 1))
    
    # Concatenate the coordinates and other parameters to form the slab
    slab = np.hstack((coords, other_params_repeated))
    
    return slab

# Unlike in the FORTRAN version, where the basis_atoms x, y and z correspond to cartesian coordinates,
# here n_1, n_2 and n_3 correspond to multiple of the lattice vectors a_1, a_2 and a_3, representing reduced coordinates
# So A is the matrix of a_1, a_2 and a_3 which is necessary to return the correct TPTS array
def extract_atom_vectors(basis_atoms: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract basis atoms' coordinates
    ns = basis_atoms[:, :3]
    TPTS = np.dot(ns,A)
    # Extract the other information
    atom_types = basis_atoms[:, 3]
    E_0 = basis_atoms[:, 4]
    U = basis_atoms[:, 5]
    n_bar = basis_atoms[:, 6]
    B = basis_atoms[:, 7:10]
    Λ = basis_atoms[:, 10]

    return TPTS, atom_types, E_0, U, n_bar, B, Λ
    