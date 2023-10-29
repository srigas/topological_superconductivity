"""
The hamiltonian.py file contains all the functions necessary for the construction of the Hamiltonian matrix, to be diagonalized
as the solution of the BdG Equations.
"""

import numpy as np

# This function calculates the hopping element constants, t_0
# It is far from optimized, however in this case readability is more important than efficiency,
# because it is a function that runs once, not through every self-consistency cycle
def hopping_consts(hop_mat: np.ndarray, N_b: int, atom_types: np.ndarray, TPTS: np.ndarray, RPTS: np.ndarray, R_0: float) -> np.ndarray:
    N_unique = hop_mat.shape[0]
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
            # Get the type of atom for the j atom
            Jatom = type_IJ[i,j]
            # Get the corresponding hopping constant
            const = t_0[Iatom,Jatom]
            # Finally, the hopping element
            t[i,j] = -const*np.exp(-np.linalg.norm(rpoint)/R_0)
        
    return t