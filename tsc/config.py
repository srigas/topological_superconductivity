"""
The config.py file ensures the loading of the configurations in the form of a class.
It also defines other helpful stuff, either separate from config values, or by using the acquired config values.
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

class Config:
    # lattice constant
    ALAT: float = 1.0
    # Bravais lattice
    latt = "SC"
    # k-points for the k-space mesh
    N_x: int = 20
    N_y: int = 20
    N_z: int = 20
    # cutoff distance to determine the nearest-neighbours
    R_max: float = 1.1
    # hopping element exponential constant
    R_0: float = 1.0
    # Create a (2NCELLS+1)^3 mini-cube to determine neighbours
    NCELLS: int = 1
    # Temperature
    T: float = 0.0
    # Initial value for μ
    mu_0: float = 0.0
    # Initial value for charges
    q_0: float = 0.8
    # Initial value for Δ
    D_0: complex = 0.3 + 0.0j
    # Mixing factor for charges
    mix_q: float = 0.1
    # Mixing factor for Δ
    mix_D: float = 0.1
    # Points for DoS
    NUME: int = 1000
    # Lorentz Gamma broadening
    l: float = 0.1
    # Threshold for metal self-consistency
    e_metal: float = 1e-4
    # Threshold for superconductor self-consistency
    e_sc: float = 1e-5

    # --------------------------------------------------------------
    # follow-ups using these values
    # --------------------------------------------------------------
    
    # Get lattice vectors
    a_1, a_2, a_3 = get_bravais(latt)
    a_1, a_2, a_3 = ALAT*a_1, ALAT*a_2, ALAT*a_3

    # Get NUMK
    NUMK: int = N_x * N_y * N_z

    # Get the Pauli matrices
    s_0, s_1, s_2, s_3 = get_pauli()
    