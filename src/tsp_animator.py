import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.tsp_utils import TSPUtils


class TSPAnimator:
    """
    Class to visualize the progress of the TSP Genetic Algorithm.

    Generates animated plots showing the evolution of the path over generations.
    """

    def __init__(self, cities, history, algorithm_name="Genetic Algorithm",
                 output_dir="../assets/Genetic/"):
        """
        Initialize the TSPAnimator.

        Args:
            cities (np.ndarray): Fixed array of city coordinates, shape (n, 2).
            history (list[np.ndarray]): History of best paths, each shape (n, 2).
            algorithm_name (str, optional): Name of the algorithm (default: "Genetic Algorithm").
            output_dir (str, optional): Directory to save the output GIF.
        """
        self.cities = cities
        self.history = history
        self.algorithm_name = algorithm_name
        self.output_dir = output_dir

        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        (self.line,) = self.ax.plot([], [], "b-", marker="o", markerfacecolor="red", markersize=5)
        self.text = self.ax.text(
            0.05,
            0.85,
            "",
            transform=self.ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    def _update(self, frame):
        """Update the plot for a single animation frame."""
        if frame < len(self.history):
            path = self.history[frame]
            x, y = path[:, 0], path[:, 1]
            self.line.set_data(x, y)

            distance = TSPUtils.evaluate_individual(path)
            self.text.set_text(
                f"Generation {frame + 1}/{len(self.history)}\n"
                f"Best Path Distance: {distance:.2f}"
            )
        return self.line, self.text

    def save(self):
        """Save the animation as a GIF file."""
        # Plot the base city locations
        self.ax.plot(self.cities[:, 0], self.cities[:, 1], "go", markersize=8)

        # Axis configuration
        self.ax.set_xlim(np.min(self.cities[:, 0]) - 5, np.max(self.cities[:, 0]) + 5)
        self.ax.set_ylim(np.min(self.cities[:, 1]) - 5, np.max(self.cities[:, 1]) + 5)
        self.ax.set_title(f"TSP Optimization using {self.algorithm_name}")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Build the animation
        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.history) + 20,
            blit=True,
            interval=400,
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        files_count = len(os.listdir(self.output_dir))
        output_path = os.path.join(self.output_dir, f"tsp_{files_count}.gif")

        anim.save(output_path, writer="pillow", fps=10)
        plt.close(self.fig)
