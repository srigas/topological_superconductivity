"""
The plotting.py file contains helper functions (e.g. for plotting) which are utilized solely for the purposes of documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The following are functions that are used to have a centralized approach to how 
# the plots look and also be able to change them universally
# ---------------------------------------------------------------------------------------
def set_custom_grid():
    # Use a grid, with specific background and grid color
    sns.set_style("darkgrid", {"axes.facecolor": "#3b3b3b", "grid.color": "#525252"})

def set_custom_legend():
    # Modify the legend properly
    legend = plt.legend()
    plt.setp(legend.get_texts(), color='white')
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
    set_custom_legend()
    
    plt.show()
    return
