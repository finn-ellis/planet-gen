import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union

class DensityField:
    """Class for handling 2D density fields with multiple component types per cell."""
    
    def __init__(self, width: int, height: int, component_types: List[str], component_masses: Optional[Dict[str, float]] = None):
        """
        Initialize a density field.
        
        Args:
            width: Width of the field in cells
            height: Height of the field in cells
            component_types: List of component names/types that can be present in cells
            component_masses: Dictionary mapping component types to their mass values (defaults to 1.0 for each)
        """
        self.width = width
        self.height = height
        self.component_types = component_types
        
        # Initialize component masses (default to 1.0 if not specified)
        self.component_masses = {}
        for comp in component_types:
            if component_masses and comp in component_masses:
                self.component_masses[comp] = component_masses[comp]
            else:
                self.component_masses[comp] = 1.0
        
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
    
    def calculate_center_of_mass(self) -> Tuple[float, float]:
        """
        Calculate the center of mass of the density field based on component masses and densities.
        
        Returns:
            List of [x, y] coordinates of the center of mass
        """
        total_weighted_x = 0.0
        total_weighted_y = 0.0
        total_mass = 0.0
        
        for y in range(self.height):
            for x in range(self.width):
                cell_mass = 0.0
                for i, component in enumerate(self.component_types):
                    density = self.field[y, x, i]
                    if density > 0:
                        # Calculate mass contribution: density percentage * component mass
                        component_mass = (density / 100.0) * self.component_masses[component]
                        cell_mass += component_mass
                
                # Add weighted contributions to center of mass
                total_weighted_x += x * cell_mass
                total_weighted_y += y * cell_mass
                total_mass += cell_mass
        
        # Avoid division by zero if field is empty
        if total_mass == 0:
            return [self.width / 2, self.height / 2]
        
        # Calculate center of mass coordinates
        com_x = total_weighted_x / total_mass
        com_y = total_weighted_y / total_mass
        
        return (com_x, com_y)
    
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
    
    def step_simulation(self, attraction_rate: float = 0.05, component_mobility: Optional[Dict[str, float]] = None) -> None:
        """
        Step the simulation by attracting components toward the center of the field.
        
        Args:
            attraction_rate: Base rate of density transfer (0-1)
            component_mobility: Optional dictionary mapping component types to mobility multipliers
                                (higher values mean the component flows more easily)
        """
        # Use the fixed center of the field as attraction point
        (center_x, center_y) = self.calculate_center_of_mass()
        
        # Initialize mobility multipliers for each component (default to 1.0)
        mobility = {}
        for comp in self.component_types:
            if component_mobility and comp in component_mobility:
                mobility[comp] = component_mobility[comp]
            else:
                mobility[comp] = 1.0
        
        # Create a copy of the current field to calculate from
        old_field = self.field.copy()
        
        # Define possible move directions (8-way connectivity including diagonals)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Process each cell in the field
        for y in range(self.height):
            for x in range(self.width):
                # Skip empty cells
                if np.sum(old_field[y, x, :]) <= 0:
                    continue
                
                # Calculate distance to center and vector toward center
                dx = center_x - x
                dy = center_y - y
                dist_to_center = np.sqrt(dx*dx + dy*dy)
                
                # Skip if we're already at the center
                if dist_to_center < 1.0:
                    continue
                
                # Normalize direction vector
                if dist_to_center > 0:
                    dx /= dist_to_center
                    dy /= dist_to_center
                
                # For each component in this cell
                for i, component in enumerate(self.component_types):
                    # Skip if no density of this component
                    if old_field[y, x, i] <= 0:
                        continue
                    
                    # Calculate density to distribute (based on component mobility and distance)
                    component_rate = attraction_rate * mobility[component]
                    density_to_distribute = old_field[y, x, i] * component_rate
                    
                    # Track how much we've distributed
                    total_distributed = 0
                    weights = []
                    valid_neighbors = []
                    
                    # Find neighbors closer to center
                    for dir_x, dir_y in directions:
                        nx, ny = x + dir_x, y + dir_y
                        
                        # Skip if out of bounds
                        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                            continue
                        
                        # Calculate if this neighbor is closer to the center
                        new_dist = np.sqrt((center_x - nx)**2 + (center_y - ny)**2)
                        
                        # Only consider neighbors that move us closer to the center
                        if new_dist < dist_to_center:
                            # Prioritize cells that are much closer to center (stronger weight)
                            dist_improvement = dist_to_center - new_dist
                            
                            # Normalize the direction vector
                            length = np.sqrt(dir_x**2 + dir_y**2)
                            dir_x /= length
                            dir_y /= length

                            # Calculate alignment with direction to center (dot product)
                            alignment = (dx * dir_x + dy * dir_y)
                            
                            # Only move in directions aligned with path to center
                            if alignment > 0:
                                # Weight based on both alignment and distance improvement
                                weight = alignment * dist_improvement
                                weights.append(weight)
                                valid_neighbors.append((nx, ny))
                    
                    # If we have valid neighbors, distribute density
                    if valid_neighbors and sum(weights) > 0:
                        # Normalize weights
                        weights = [w / sum(weights) for w in weights]
                        
                        # Distribute density to valid neighbors
                        for (nx, ny), weight in zip(valid_neighbors, weights):
                            # Calculate how much density to move to this neighbor
                            amount = density_to_distribute * weight
                            
                            # Calculate available space in the target cell
                            current_total = np.sum(self.field[ny, nx, :])
                            available_space = 100 - current_total
                            
                            # Cap the amount to the available space
                            amount = min(amount, available_space)
                            
                            # Move density from source to target
                            self.field[y, x, i] -= amount
                            self.field[ny, nx, i] += amount
                            total_distributed += amount
                        
                        # If we couldn't distribute all intended density, keep it in the original cell
                        if total_distributed < density_to_distribute:
                            self.field[y, x, i] += (density_to_distribute - total_distributed)
    
    def animate_simulation(self, steps: int, attraction_rate: float = 0.05, 
                           component_mobility: Optional[Dict[str, float]] = None,
                           interval: int = 100, cmap_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Animate the simulation by showing each step.
        
        Args:
            steps: Number of simulation steps to run
            attraction_rate: Rate of density attraction per step
            component_mobility: Optional dictionary of component mobility multipliers
            interval: Milliseconds between animation frames
            cmap_dict: Color map for components
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Setup the initial plot
        if cmap_dict is None:
            default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            cmap_dict = {comp: default_colors[i % len(default_colors)] 
                        for i, comp in enumerate(self.component_types)}
        
        # Create a function to update the plot for each frame
        def update(frame):
            # Step the simulation
            self.step_simulation(attraction_rate, component_mobility)
            
            # Clear the axis for the new frame
            ax.clear()
            
            # Plot the current state
            self.plot_composite(cmap_dict=cmap_dict, ax=ax)
            
            # Calculate and display center of mass
            com_x, com_y = self.calculate_center_of_mass()
            ax.plot(com_x, com_y, 'ko', markersize=8)
            ax.text(com_x + 1, com_y + 1, "Center of Mass", color='black', fontweight='bold')
            
            ax.set_title(f'Simulation Step {frame+1}')
            
        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim


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
    field = DensityField(50, 50, ['Air', 'Water', 'Earth', 'Fire'], 
                          component_masses={'Air': 1, 'Water': 2, 'Earth': 3, 'Fire': 1})
    
    # Create an asymmetric field to see movement
    # for y in range(20, 40):
    #     for x in range(10, 30):
    #         field.set_cell_density(x, y, {'Earth': 70, 'Water': 20})
    
    # for y in range(10, 25):
    #     for x in range(25, 45):
    #         field.set_cell_density(x, y, {'Fire': 60, 'Air': 30})
    field.set_all_cell_densities({'Air': 50})
    
    # Set different mobility rates
    mobility = {'Air': 1.0, 'Water': 1.0, 'Earth': 1.0, 'Fire': 1.0}
    
    # Animate the simulation
    field.animate_simulation(
        steps=50, 
        attraction_rate=0.05,
        component_mobility=mobility,
        cmap_dict={'Air': 'white', 'Water': 'blue', 'Earth': 'brown', 'Fire': 'red'}
    )

if __name__ == "__main__":
    # example_usage()
    test_simulation()