import numpy as np

from src.algorithms.algorithm import Algorithm

class HillClimbing(Algorithm):
    def __init__(self, lower_bound, upper_bound, function, iterations=10_000):
        super().__init__(lower_bound, upper_bound, function, iterations)

    def run(self):
        history = []

        x = np.random.uniform(self.lower_bound, self.upper_bound)
        y = np.random.uniform(self.lower_bound, self.upper_bound)
        point = [x, y]
        z = self.function(np.array(point))