import numpy as np

from algorithm import Algorithm

class BlindSearch(Algorithm):
    def __init__(self, lower_bound, upper_bound, function, iterations=1000):
        super().__init__(lower_bound, upper_bound, function, iterations)

    def run(self):
        history = []

        x = np.random.uniform(self.lower_bound, self.upper_bound) 
        y = np.random.uniform(self.lower_bound, self.upper_bound)
        point = [x, y]
        z = function(np.array(point))
        min_z = z
        history.append((point[0], point[1], z))

        while min_z != 0 and k < self.iterations:
            x = np.random.uniform(self.lower_bound, self.upper_bound)
            y = np.random.uniform(self.lower_bound, self.upper_bound)
            point = np.array([x, y])
            z = self.function(point)
            k += 1

            if z < min_z:
                min_z = z
                history.append((point[0], point[1], z))
                k = 0 

        return history
