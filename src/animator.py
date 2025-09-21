import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from solution import Solution

class Animator:
    def __init__(self, solution: Solution):
        self.solution = solution
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points = []
        self.params = [np.arange(self.solution.lower_bound, self.solution.upper_bound, self.solution.step) for _ in range(self.solution.dimension)]
        self.params = np.meshgrid(*self.params)

    def plot(self):
        x, y = self.params
        z = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = self.solution.function(np.array([x[i, j], y[i, j]]))
        
        self.ax.plot_surface(x, y, z, cmap='magma', edgecolor='none', alpha=0.9)
        self.ax.set_title(f"{str.capitalize(self.solution.function.__name__)} function")

    def show(self):
        self.plot

    def animate(self, index):
        print(f"Animating step #{index}")

        for point in self.points:
            point.set_data([], [])
            point.set_3d_properties([])
        self.points = []

        if index < len(self.solution.history) and self.solution.history[index]:
            for p in self.solution.history[index]:
                if len(p) == 3:
                    x_val, y_val, z_val = p
                else:
                    x_val, y_val = p
                    z_val = self.solution.function(np.array([x_val, y_val]))
                
                point_plot, = self.ax.plot([x_val], [y_val], [z_val], 'ro', markersize=10)
                self.points.append(point_plot)

        return self.points

    def save(self):
        self.plot()
        anim = FuncAnimation(self.fig, self.animate, frames=min(len(self.solution.history), 50), interval=250)
        output_path = f"../assets/{self.solution.algorithm.__class__.__name__}/{self.solution.function.__name__}.gif"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anim.save(output_path, writer='pillow')
        plt.close(self.fig)