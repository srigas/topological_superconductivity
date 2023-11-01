"""
The plotting.py file contains helper functions (e.g. for plotting) which are utilized solely for the purposes of documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from typing import List

# The following are functions that are used to have a centralized approach to how 
# the plots look and also be able to change them universally
# ---------------------------------------------------------------------------------------
def set_custom_grid():
    # Use a grid, with specific background and grid color
    sns.set_style("darkgrid", {"axes.facecolor": "#3b3b3b", "grid.color": "#525252"})
# ---------------------------------------------------------------------------------------

# Function to plot a total DoS diagram
def plot_tdos(Es: np.ndarray, TDoS: np.ndarray):
    # Setup the grid
    set_custom_grid()
    
    # Setup figure, set dimensions
    plt.figure(figsize=(10, 4))
    # Draw the DoS as a lineplot
    sns.lineplot(x=Es, y=TDoS, label='Density of States', color='#d957ce')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.title('Density of States Plot')

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
