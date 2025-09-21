import numpy as np

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.function = function
        self.algorithm = algorithm(lower_bound, upper_bound, function)
        self.history = []

    def find_minimum(self):
        best_value = float('inf')
        self.history = []

        all_points = self.algorithm.run()

        for point in all_points:
            x, y, z = point
            if z < best_value:
                best_value = z
                self.history.append([point])

        return best_value