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

def plot_contour(Es: np.ndarray):
    # Setup the grid
    set_custom_grid()

    # Setup figure, set dimensions
    plt.figure(figsize=(8, 4))

    # Draw the points as a scatter plot, where x = Re(E) and y = Im(E)
    Re_E, Im_E = Es.real, Es.imag
    plt.scatter(Re_E, Im_E, s = 2.0, color='#d957ce')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Energy Mesh Plot')
    
    plt.show()

    return