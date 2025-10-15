import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.solution import Solution


class Animator:
    """
    A class to visualize the optimization process of an algorithm.

    The Animator generates 3D plots of a given function's surface and
    overlays the optimization algorithm's search history. It supports
    displaying static plots, animating the optimization steps, and
    saving the animation as a GIF file.

    Attributes:
        solution (Solution): The solution object containing the algorithm,
                             function, search bounds, and optimization history.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axis for plotting.
        points (list): List of plotted points representing optimization steps.
        params (list of np.ndarray): Meshgrid of parameters (x, y) for surface plotting.
        _surface (matplotlib.surface): The plotted function surface.
    """

    def __init__(self, solution: Solution):
        """
        Initialize the Animator with a given solution.

        Args:
            solution (Solution): The solution containing the algorithm,
                                 function, and optimization history.
        """
        self.solution = solution
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.points = []

        # Create parameter meshgrid for surface plotting
        self.params = [
            np.arange(self.solution.lower_bound, self.solution.upper_bound, self.solution.step)
            for _ in range(self.solution.dimension)
        ]
        self.params = np.meshgrid(*self.params)
        self._surface = None

    def plot(self):
        """
        Plot the function surface on the 3D axis.

        Creates a surface plot of the objective function using meshgrid points.
        """
        x, y = self.params
        z = np.zeros_like(x)

        # Evaluate function values for each grid point
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = self.solution.function(np.array([x[i, j], y[i, j]]))

        self._surface = self.ax.plot_surface(
            x, y, z, cmap="magma", edgecolor="none", alpha=0.3
        )
        self.ax.set_title(f"{str.capitalize(self.solution.function.__name__)} function")

    def show(self):
        """
        Display the 3D surface plot with the optimization history.
        """
        self.plot()
        plt.show()

    def animate(self, index):
        """
        Animate one frame of the optimization process.

        Args:
            index (int): The current frame index.

        Returns:
            list: A list of matplotlib artist objects representing plotted points.
        """
        print(f"Animating step #{index}")

        # Remove previously plotted point artists cleanly
        for artist in self.points:
            try:
                artist.remove()
            except Exception:
                # fallback: try to clear data
                try:
                    artist.set_data([], [])
                    artist.set_3d_properties([])
                except Exception:
                    pass
        self.points = []

        # Plot new points from history (history entries are lists of points)
        if index < len(self.solution.history) and self.solution.history[index]:
            for p in self.solution.history[index]:
                # Accept sequences or numpy arrays
                try:
                    length = len(p)
                except Exception:
                    # if p is not sized, skip
                    continue

                if length == 3:
                    x_val, y_val, z_val = p
                elif length == 2:
                    x_val, y_val = p
                    z_val = self.solution.function(np.array([x_val, y_val]))
                else:
                    # unexpected shape, skip
                    continue

                point_plot, = self.ax.plot([x_val], [y_val], [z_val], "ro", markersize=6)
                self.points.append(point_plot)

        return self.points

        return self.points

    def save(self):
        """
        Save the optimization process as an animated GIF.

        The animation shows the function surface with the algorithm's search
        history overlaid step by step.

        Output:
            A GIF file saved under:
            `../assets/{algorithm_name}/{function_name}.gif`
        """
        self.plot()
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=min(len(self.solution.history), 50),
            interval=250,
        )

        output_path = f"../assets/{self.solution.algorithm.__class__.__name__}/{self.solution.function.__name__}.gif"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anim.save(output_path, writer="pillow")
        plt.close(self.fig)
