import numpy as np

from src.algorithms.algorithm import Algorithm

class HillClimbing(Algorithm):
    """
    Hill Climbing optimization algorithm.

    This algorithm iteratively explores the search space by generating
    neighbors around a current point and moving to a neighbor if it
    improves the objective function. The process repeats until no
    improvement is found for a number of iterations or the maximum
    number of iterations is reached.
    """
    def __init__(
            self,
            lower_bound,
            upper_bound,
            function,
            iterations=10_000,
            neighbors=15,
            sigma =.6
        ):
        """
        Initialize the HillClimbing algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): Function to minimize. Must accept a NumPy array.
            iterations (int, optional): Maximum number of iterations without improvement (default is 10,000).
            neighbors (int, optional): Number of neighbor candidates to generate in each iteration (default is 15).
            sigma (float, optional): Standard deviation of the Gaussian distribution (default is 0.6).
        """
        super().__init__(lower_bound, upper_bound, function, iterations)
        self.neighbors = neighbors
        self.sigma = sigma

    def _generate(self, point):
        """
        Generate neighbor points around a given point using Gaussian perturbation.

        Args:
            point (np.ndarray): Current point in the search space.

        Returns:
        list of np.ndarray: A list of neighbor points within the specified bounds.
        """
        neighbors = []

        for _ in range(self.neighbors):
            # Add Gaussian noise around the current point
            neighbor = np.random.normal(point, self.sigma, size=len(point))
            # Ensure the neighbor lies within the search space
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            neighbors.append(neighbor)
        return neighbors


    def run(self):
        """
        Run the hill climbing optimization process.

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

        z = self.function(point)
        min_z = z # Current best function value
        history.append((point[0], point[1], z))
        k = 0 # Counter for iterations since last improvement

        # Run the HillClimbing until a solution with z == 0 is found
        # or until the iteration limit is reached
        while min_z != 0 and k < self.iterations:
            neighbors = self._generate(point)
            improved = False

            # Evaluate neighbors and move to the first improving one
            for neighbor in neighbors:
                value = self.function(neighbor)
                # If the new point is better, update best result
                if value < min_z:
                    point = neighbor
                    min_z = value
                    history.append((point[0], point[1], value))
                    k = 0
                    improved = True
                    break
            if not improved:
                k += 1

        return history