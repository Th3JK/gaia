import numpy as np

from src.algorithms.algorithm import Algorithm


class BlindSearch(Algorithm):
    """
    Blind Search optimization algorithm.

    This algorithm searches for the minimum of a given function by randomly
    sampling points within the given bounds. It records improvements and
    terminates when either:
      - a function value of 0 is found, or
      - the maximum number of iterations without improvement is reached.
    """
    def __init__(
            self,
            lower_bound,
            upper_bound,
            function,
            iterations=10_000
        ):
        """
        Initialize the BlindSearch algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): Function to minimize. Must accept a NumPy array.
            iterations (int, optional): Maximum number of iterations without improvement.
                                         Defaults to 10,000.
        """
        super().__init__(lower_bound, upper_bound, function, iterations)

    def run(self):
        """
        Run the blind search optimization process.

        Returns:
            list of tuples: A history of improvements found during the search.
                            Each entry is a tuple (x, y, z) where:
                              - x, y are the coordinates of the sampled point
                              - z is the function value at that point
        """
        # Stores the history of best points found (x, y, function_value)
        history = []

        # Generate a random starting point within the bounds
        point = np.array([
            np.random.uniform(self.lower_bound, self.upper_bound),
            np.random.uniform(self.lower_bound, self.upper_bound)
        ])

        # Evaluate the function at the starting point
        z = self.function(np.array(point))
        min_z = z # Current best function value
        history.append((point[0], point[1], z))
        k = 0 # Counter for iterations since last improvement

        # Run the blind/random search until a solution with z == 0 is found
        # or until the iteration limit is reached
        while min_z != 0 and k < self.iterations:
            # Generate a new random point
            point = np.array([
                np.random.uniform(self.lower_bound, self.upper_bound),
                np.random.uniform(self.lower_bound, self.upper_bound)
            ])

            # Evaluate function at new point
            z = self.function(point)
            k += 1

            # If the new point is better, update best result
            if z < min_z:
                min_z = z
                history.append((point[0], point[1], z))
                k = 0 

        return history
