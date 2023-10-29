"""
The doc_plots.py file contains helper functions for plotting which are utilized for the purposes of documentation.
"""

import numpy as np
import plotly.graph_objects as go

# Helper function for documentation that plots the lattice sites
def plot_lattice(RPTS: np.ndarray):
    
    # Splitting the vectors into x, y, z coordinates
    x_coords = RPTS[:, 0]
    y_coords = RPTS[:, 1]
    z_coords = RPTS[:, 2]
    
    # Creating the scatter plot for original vectors
    scatter = go.Scatter3d(x=x_coords, y=y_coords, z=z_coords,
                           mode='markers',
                           marker=dict(size=5, color='blue', symbol='cross'))
    
    fig = go.Figure(data=scatter)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
    return

# Helper function for documentation that plots the basis atoms on top of the lattice sites
def plot_atoms(RPTS: np.ndarray, TPTS: np.ndarray):
    
    # Lattice sites
    lat_x, lat_y, lat_z = RPTS[:, 0], RPTS[:, 1], RPTS[:, 2]

    # Basis atoms - type 1
    B1PTS = RPTS + TPTS[0]
    b_x_1, b_y_1, b_z_1 = B1PTS[:, 0], B1PTS[:, 1], B1PTS[:, 2]

    # Basis atoms - type 2
    B2PTS = RPTS + TPTS[1]
    b_x_2, b_y_2, b_z_2 = B2PTS[:, 0], B2PTS[:, 1], B2PTS[:, 2]
    
    # The lattice sites
    lattice = go.Scatter3d(x=lat_x, y=lat_y, z=lat_z,
                           mode='markers',
                           marker=dict(size=5, color='blue', symbol='cross'))

    # The first type of atom
    atom_1 = go.Scatter3d(x=b_x_1, y=b_y_1, z=b_z_1,
                           mode='markers',
                           marker=dict(size=3, color='blue', symbol='circle'))

    # The second type of atom
    atom_2 = go.Scatter3d(x=b_x_2, y=b_y_2, z=b_z_2,
                           mode='markers',
                           marker=dict(size=3, color='red', symbol='circle'))
    
    # Combining all scatter plots in one figure
    fig = go.Figure(data=[lattice, atom_1, atom_2])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
    return