import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union

class DensityField:
    """Class for handling 2D density fields with multiple component types per cell."""
    
    def __init__(self, width: int, height: int, component_types: List[str]):
        """
        Initialize a density field.
        
        Args:
            width: Width of the field in cells
            height: Height of the field in cells
            component_types: List of component names/types that can be present in cells
        """
        self.width = width
        self.height = height
        self.component_types = component_types
        
        # Initialize empty field - a 2D grid where each cell contains a dictionary of component densities
        self.field = np.zeros((height, width, len(component_types)))
    
    def set_cell_density(self, x: int, y: int, component_values: Dict[str, float]) -> None:
        """
        Set density values for a specific cell.
        
        Args:
            x: X-coordinate of the cell
            y: Y-coordinate of the cell
            component_values: Dictionary mapping component types to density values (0-100)
        """
        total = sum(component_values.values())
        if total > 100:
            raise ValueError(f"Total density exceeds 100%: {total}%")
            
        for component, value in component_values.items():
            if component not in self.component_types:
                raise ValueError(f"Unknown component type: {component}")
            idx = self.component_types.index(component)
            self.field[y, x, idx] = value
    
    def get_cell_density(self, x: int, y: int) -> Dict[str, float]:
        """Get density values for a specific cell."""
        result = {}
        for i, component in enumerate(self.component_types):
            value = self.field[y, x, i]
            if value > 0:
                result[component] = value
        return result
    
    def generate_random_field(self, max_total: float = 95.0) -> None:
        """Generate a random density field for testing."""
        for y in range(self.height):
            for x in range(self.width):
                # For each cell, randomly assign densities that sum to less than max_total
                values = {}
                remaining = max_total
                for i, component in enumerate(self.component_types[:-1]):
                    if remaining <= 0:
                        break
                    value = np.random.uniform(0, remaining)
                    if value > 0:
                        values[component] = value
                    remaining -= value
                
                # Ensure last component gets some density if there's room
                if remaining > 0 and np.random.random() > 0.3:
                    values[self.component_types[-1]] = np.random.uniform(0, remaining)
                
                if values:
                    self.set_cell_density(x, y, values)
    
    def set_all_cell_densities(self, component_values: Dict[str, float]) -> None:
        """Generate a random density field for testing."""
        for y in range(self.height):
            for x in range(self.width):
                self.set_cell_density(x, y, component_values)
    
    def plot_composite(self, cmap_dict: Optional[Dict[str, str]] = None, 
                       ax: Optional[plt.Axes] = None, show_empty: bool = False) -> plt.Figure:
        """
        Plot the density field as a composite image where colors represent component mixtures.
        
        Args:
            cmap_dict: Dictionary mapping component types to color names
            ax: Matplotlib axis to plot on (creates new figure if None)
            show_empty: Whether to show cells with no density
            
        Returns:
            The matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
            
        # Default colors if not specified
        if cmap_dict is None:
            default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            cmap_dict = {comp: default_colors[i % len(default_colors)] 
                        for i, comp in enumerate(self.component_types)}
        
        # Create an RGB image to represent the densities
        rgb_image = np.zeros((self.height, self.width, 3))
        
        for y in range(self.height):
            for x in range(self.width):
                densities = self.get_cell_density(x, y)
                if not densities and not show_empty:
                    continue
                    
                # Calculate weighted color based on component densities
                total = sum(densities.values())
                for component, value in densities.items():
                    color = plt.cm.colors.to_rgb(cmap_dict[component])
                    # Scale color by density proportion
                    weight = value / 100.0  # Normalize to 0-1 range
                    rgb_image[y, x, :] += np.array(color) * weight
        
        # Clip to ensure valid RGB values
        rgb_image = np.clip(rgb_image, 0, 1)
        
        # Plot the image
        ax.imshow(rgb_image, origin='lower')
        ax.set_title('Composite Density Field')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap_dict[comp], label=f'{comp}')
                          for comp in self.component_types]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        return fig
    
    def plot_components(self, cmap: str = 'viridis') -> plt.Figure:
        """
        Plot each component density as a separate subplot.
        
        Args:
            cmap: Colormap name to use for the plots
            
        Returns:
            The matplotlib figure
        """
        n_components = len(self.component_types)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_components == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, component in enumerate(self.component_types):
            if i < len(axes):
                ax = axes[i]
                component_data = self.field[:, :, i]
                im = ax.imshow(component_data, origin='lower', cmap=cmap, vmin=0, vmax=100)
                ax.set_title(f'{component} Density')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='Density %')
        
        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        return fig


def example_usage():
    """Example demonstrating the use of the DensityField class."""
    # Create a density field with 3 component types
    field = DensityField(20, 15, ['Water', 'Rock', 'Metal', 'Gas'])
    
    # Generate random data
    field.generate_random_field(max_total=95.0)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    field.plot_composite(
        cmap_dict={'Water': 'blue', 'Rock': 'brown', 'Metal': 'gray', 'Gas': 'lightskyblue'},
        ax=plt.gca()
    )
    
    plt.subplot(1, 2, 2)
    # Set some specific cells to demonstrate
    field.set_cell_density(5, 5, {'Water': 70, 'Rock': 20})
    field.set_cell_density(5, 6, {'Rock': 40, 'Metal': 30})
    field.set_cell_density(6, 5, {'Water': 30, 'Gas': 60})
    field.set_cell_density(6, 6, {'Metal': 90})
    
    field.plot_components()
    
    plt.tight_layout()
    plt.show()

def plot_field(field: DensityField):
    field.plot_composite(
        cmap_dict={'Air': 'white', 'Water': 'blue', 'Earth': 'brown', 'Fire': 'red'},
        ax=plt.gca()
    )
    plt.tight_layout()
    plt.show()

def test_simulation():
    field = DensityField(50, 50, ['Air', 'Water', 'Earth', 'Fire'])

    field.set_all_cell_densities({'Air': 50})

    plot_field(field)

if __name__ == "__main__":
    # example_usage()
    test_simulation()