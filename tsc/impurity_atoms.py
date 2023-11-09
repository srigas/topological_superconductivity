"""
The impurity_atoms.py file holds functions that generate the required parameters for the problem's impurity atoms.
The two main functions [1] create a single basis atom or [2] create a chain of magnetic impurities in a spin-helix
configuration. Additional functions can be added to create different types of configurations, which may not be as universal.
Also included are auxiliary functions that help extract arrays.
"""

import numpy as np
from typing import Tuple

# Function that takes an array of basis atom indices and returns the corresponding basis atoms info
def extract_prev_vals(indices: np.ndarray, atoms:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Extract relevant info
    E_0_prev = atoms[indices, 4]
    U_prev = atoms[indices, 5]
    n_bar_prev = atoms[indices, 6]
    B_prev = atoms[indices, 7:10]
    Λ_prev = atoms[indices, 10]

    return E_0_prev, U_prev, n_bar_prev, B_prev, Λ_prev

# Function that returns a single impurity atom's info
def single_imp(N_x: int = 0, N_y: int = 0, N_z: int = 0, atom_index: int = 0, 
               E_0_new: float = 0.0, U_new: float = 0.0, n_bar_new: float = 1.0, 
               B_x_new: float = 0.0, B_y_new: float = 0.0, B_z_new: float = 0.0, Λ_new: float = 0.5) -> np.ndarray:
    
    imp = np.array([[N_x, N_y, N_z, atom_index, E_0_new, U_new, n_bar_new, B_x_new, B_y_new, B_z_new, Λ_new]])
    return imp

# Function that extracts new info for single impurity atoms or arrays thereof
def extract_new_vals(imps: np.ndarray, TPTS: np.ndarray, a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Make IPTS array by multiplying lattice vectors by the corresponding ints
    IPTS = np.zeros((imps.shape[0], 3))
    IPTS = imps[:, 0, np.newaxis]*a_1 + imps[:, 1, np.newaxis]*a_2 + imps[:, 2, np.newaxis]*a_3
    # Extract basis atom indices
    indices = imps[:, 3].astype(int)
    # Now add the corresponding basis atoms
    IPTS += TPTS[indices,:]

    # Extract new values for E_0, U, n_bar, B, Λ
    E_0_new = imps[:, 4]
    U_new = imps[:, 5]
    n_bar_new = imps[:, 6]
    B_new = imps[:, 7:10]
    Λ_new = imps[:, 10]

    return IPTS, indices, E_0_new, U_new, n_bar_new, B_new, Λ_new

# Function that returns a magnetic chain's info with a spin-helix configuration
def mag_chain(N_imp: int, TPTS: np.ndarray, θ: float = 2.0*np.pi/3.0, B_0: np.ndarray = np.array([2.0, 0.0, 0.0]), 
              Rot_axis: np.ndarray = np.array([0.0, 1.0, 0.0]), atom_index: int = 0,
              chain_axis: str = 'x') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert chain_axis in ('x', 'y', 'z'), "Invalid chain axis. Choose from 'x', 'y', or 'z'"

    # Calculate sines and cosines
    cos, sin = np.cos(θ), np.sin(θ)
    
    # Setup the rotation matrix
    R = np.zeros((3,3))
    
    R[0,0] = cos + Rot_axis[0]**2*(1.0-cos)
    R[0,1] = Rot_axis[0]*Rot_axis[1]*(1.0-cos) - Rot_axis[2]*sin
    R[0,2] = Rot_axis[0]*Rot_axis[2]*(1.0-cos) + Rot_axis[1]*sin

    R[1,0] = Rot_axis[0]*Rot_axis[1]*(1.0-cos) + Rot_axis[2]*sin
    R[1,1] = cos + Rot_axis[1]**2*(1.0-cos)
    R[1,2] = Rot_axis[1]*Rot_axis[2]*(1.0-cos) - Rot_axis[0]*sin
    
    R[2,0] = Rot_axis[0]*Rot_axis[2]*(1.0-cos) - Rot_axis[1]*sin
    R[2,1] = Rot_axis[1]*Rot_axis[2]*(1.0-cos) + Rot_axis[0]*sin
    R[2,2] = cos + Rot_axis[2]**2*(1.0-cos)

    # Setup the B_new array
    B_new = np.zeros((N_imp,3))
    B_new[0,:] = B_0

    for i in range(1,N_imp):
        B_new[i,:] = R@B_new[i-1,:]

    # Initialize to None
    xs = ys = zs = None

    # one coordinate must be increasing from a starting value
    if chain_axis == 'x':
        xs = np.arange(TPTS[atom_index,0],TPTS[atom_index,0]+N_imp)
    elif chain_axis == 'y':
        ys = np.arange(TPTS[atom_index,1],TPTS[atom_index,1]+N_imp)
    elif chain_axis == 'z':
        zs = np.arange(TPTS[atom_index,2],TPTS[atom_index,2]+N_imp)
                    
    # the remaining two coordinates can be left constant
    if xs is None:
        xs = np.ones((N_imp))*TPTS[atom_index,0]
    if ys is None:
        ys = np.ones((N_imp))*TPTS[atom_index,1]
    if zs is None:
        zs = np.ones((N_imp))*TPTS[atom_index,2]

    # Set zero superconductivity
    Λ_new = np.zeros((N_imp))
    
    return xs, ys, zs, B_new, Λ_new
    