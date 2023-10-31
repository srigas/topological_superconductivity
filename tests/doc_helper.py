"""
The doc_helper.py file contains helper functions (e.g. for plotting) which are utilized solely for the purposes of documentation.
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

    # Basis atom
    BPTS = RPTS + TPTS[0]
    b_x, b_y, b_z = BPTS[:, 0], BPTS[:, 1], BPTS[:, 2]
    
    # The lattice sites
    lattice = go.Scatter3d(x=lat_x, y=lat_y, z=lat_z,
                           mode='markers',
                           marker=dict(size=5, color='blue', symbol='cross'))

    # The basis atoms
    atoms = go.Scatter3d(x=b_x, y=b_y, z=b_z,
                           mode='markers',
                           marker=dict(size=3, color='blue', symbol='circle'))
    
    # Combining both scatter plots in one figure
    fig = go.Figure(data=[lattice, atoms])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
    return