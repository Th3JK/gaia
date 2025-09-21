import numpy as np

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.function = function
        self.algorithm = algorithm
        self.history = []

    def find_minimum(self):
        best_value = float('inf')
        self.history = []

        all_points = self.algorithm.run()

        for points in all_points:
            current_value = self.function(np.array(points[-1]))
            if current_value < best_value:
                best_value = current_value
                self.history.append(points)

        return best_value