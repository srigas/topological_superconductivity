"""
The plotting.py file contains helper functions (e.g. for plotting) which are utilized solely for the purposes of documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

from typing import List

# The following are functions that are used to have a centralized approach to how 
# the plots look and also be able to change them universally
# ---------------------------------------------------------------------------------------
def set_custom_grid():
    # Use a grid, with specific background and grid color
    sns.set_style("darkgrid", {"axes.facecolor": "#3b3b3b", "grid.color": "#525252"})
# ---------------------------------------------------------------------------------------

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

# Function to plot a total DoS diagram
def plot_tdos(Es: np.ndarray, TDoS: np.ndarray):
    # Setup the grid
    set_custom_grid()
    
    # Setup figure, set dimensions
    plt.figure(figsize=(10, 4))
    # Draw the DoS as a lineplot
    plt.plot(Es, TDoS, label='Density of States', color='#d957ce')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.title('Density of States Plot')

    # Stylize the legend
    legend = plt.legend()
    plt.setp(legend.get_texts(), color='white')
    
    plt.show()
    return

# Function to plot the integrated number density
def plot_intnumden(E_vals_sorted: np.ndarray, intnumden: np.ndarray, μ: float):
    # Setup the grid
    set_custom_grid()
    
    # Setup figure, set dimensions
    plt.figure(figsize=(10, 4))
    # Draw the lineplot
    plt.plot(E_vals_sorted, intnumden, color='#d957ce')
    # Add the Fermi level
    plt.axvline(x=μ, color='#e0eb8f', linestyle='--', label=f'Fermi Level')
    
    plt.xlabel('Energy')
    plt.ylabel('Number of states (Normalized)')
    plt.title('Integrated Number Density')

    # Stylize the legend
    legend = plt.legend()
    plt.setp(legend.get_texts(), color='white')
    
    plt.show()
    return

# This function plots the band structure for the non-weighted problem
def plot_bands(band_vals: np.ndarray, x_for_plot: np.ndarray, plot_points_x: np.ndarray, letters: List[str], μ: float, points_str: str):
    
    # Setup the grid
    set_custom_grid()

    fig, ax = plt.subplots(figsize=(8,5))

    # Define colors for each category
    colors = ['#e38fdc', '#b18fe3', '#e0eb8f', '#f2db7e']
    labels = ['Electron ↑', 'Electron ↓', 'Hole ↑', 'Hole ↓']
    
    # Add high symmetry lines
    for x in plot_points_x:
        ax.axvline(x=x, color='#525252', linestyle='-', lw=0.5)
    
    # Plot bands
    for band in range(band_vals.shape[1]):
        category = band % 4
        color = colors[category]
        label = labels[category] if band < 4 else "" # Get the label only once
        ax.plot(x_for_plot, band_vals[:, band], color=color, label=label)
    
    # Set x-axis tick labels to high symmetry point letters
    ax.set_xticks(plot_points_x)
    ax.set_xticklabels(letters)
    
    # Draw horizontal line at Fermi energy level and create a custom legend entry
    ax.axhline(y=μ, color='#82e8dc', linestyle='--', lw=0.8)
    fermi_line, = ax.plot([], [], color='#82e8dc', linestyle='--', lw=0.8, label='Fermi level')
    
    # Create the legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Change the color of the legend text to white
    for text in legend.get_texts():
        text.set_color("white")
    
    ax.set_xlabel('')
    ax.set_ylabel('Energy')
    ax.set_title(f'Band Diagram for direction {points_str}')
    
    plt.show()
    return

# This function plots the band structure for the weighted problem
def plot_bands_weighted(atom: int, weights: np.ndarray, band_vals: np.ndarray, x_for_plot: np.ndarray, plot_points_x: np.ndarray, letters: List[str], μ: float, points_str: str):

    # Choose atom
    weights_a = weights[atom]
    
    # Setup the grid
    set_custom_grid()

    fig, ax = plt.subplots(figsize=(8,5))

    # Get the colormap
    cmap = plt.get_cmap('spring')

    # Get the normalization object based on the weight range
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    
    # Add high symmetry lines
    for x in plot_points_x:
        ax.axvline(x=x, color='#525252', linestyle='-', lw=0.5)
    
    # Plot bands
    for band in range(band_vals.shape[1]):
        # Scatter plot for each band with color based on weights
        sc = ax.scatter(x_for_plot, band_vals[:, band], c=weights_a[:, band], cmap=cmap, norm=norm, s=1)
    
    # Set x-axis tick labels to high symmetry point letters
    ax.set_xticks(plot_points_x)
    ax.set_xticklabels(letters)
    
    # Draw horizontal line at Fermi energy level and create a custom legend entry
    ax.axhline(y=μ, color='#82e8dc', linestyle='--', lw=0.8)
    fermi_line, = ax.plot([], [], color='#82e8dc', linestyle='--', lw=0.8, label='Fermi level')

    # Create the legend
    legend = ax.legend(loc='upper right')
    # Change the color of the legend text to white
    for text in legend.get_texts():
        text.set_color("white")

    # Create colorbar
    cb = plt.colorbar(sc, ax=ax)
    cax = cb.ax
    cax.set_title(f'Atom {atom+1} Weight', position=(1, 1))  # Adjust position to your liking

    
    ax.set_xlabel('')
    ax.set_ylabel('Energy')
    ax.set_title(f'Band Diagram for direction {points_str}')
    
    plt.show()
    return
