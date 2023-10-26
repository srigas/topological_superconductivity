"""
The config.py file ensures the loading of the configurations in the form of a class.
"""

class Config:
    # Define tolerance for lattice vectors' norms
    TOL: float = 1e-4
    # lattice constant
    ALAT: float = 1.0
    # Bravais vectors
    a_1, a_2, a_3 = get_vectors("SC")
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