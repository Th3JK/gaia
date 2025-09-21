import numpy as np

class Solution: 
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.parameters = np.zeros(self.dimension)
        self.evaluation_function = np.inf
        