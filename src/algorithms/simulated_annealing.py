import numpy as np

from src.algorithms.algorithm import Algorithm


class SimulatedAnnealing(Algorithm):
    """
    Simulated Annealing optimization algorithm.

    This algorithm searches for the minimum of a given function by exploring
    the search space with probabilistic acceptance of worse solutions to
    escape local minima. The probability of accepting worse solutions decreases
    over time according to a cooling schedule.
    """
    def __init__(
            self,
            lower_bound,
            upper_bound,
            function,
            iterations=1_000,
            initial_temperature=100,
            minimal_temperature=.5,
            alpha=.9,
            sigma=.6
        ):
        """
        Initialize the SimulatedAnnealing algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): Function to minimize. Must accept a NumPy array.
            iterations (int, optional): Maximum number of iterations without improvement. Defaults to 1000.
            initial_temperature (float, optional): Starting temperature. Defaults to 100.
            minimal_temperature (float, optional): Minimum temperature to stop. Defaults to 0.5.
            alpha (float, optional): Cooling rate per iteration. Defaults to 0.9.
            sigma (float, optional): Standard deviation for neighbor generation. Defaults to 0.6.
        """
        super().__init__(lower_bound, upper_bound, function, iterations)
        self.initial_temperature = initial_temperature
        self.minimal_temperature = minimal_temperature
        self.alpha = alpha
        self.sigma = sigma

    def _generate(self, point):
        """
        Generate a neighbor point using Gaussian perturbation.

        Args:
            point (np.ndarray): Current point in the search space.

        Returns:
            np.ndarray: A neighbor point within bounds.
        """
        neighbor = np.random.normal(point, self.sigma, size=len(point))
        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor

    def run(self):
        """
        Run the Simulated Annealing optimization process.

        Returns:
            list of tuples: History of points explored during the search. Each entry is
                            a tuple (x, y, z) where:
                              - x, y: coordinates of the point
                              - z: function value at that point
        """
        # Stores the history of best points found (x, y, function_value)
        history = []

        # Generate a random starting point within the bounds
        point = np.array([
            np.random.uniform(self.lower_bound, self.upper_bound),
            np.random.uniform(self.lower_bound, self.upper_bound)
        ])

        current_fitness = self.function(point)
        history.append((point[0], point[1], current_fitness))
        temperature = self.initial_temperature
        k = 0 # Counter for iterations since last improvement

        # Main loop: continue until temperature is minimal or max iterations reached
        while temperature > self.minimal_temperature and k < self.iterations:
            neighbor = self._generate(point)
            neighbor_fitness = self.function(neighbor)
            # Compute probability of accepting worse solution
            acceptance_prob = np.exp(-(neighbor_fitness - current_fitness) / temperature)

            # Accept new point if it's better or probabilistically if worse
            if neighbor_fitness < current_fitness:
                point = neighbor
                current_fitness = neighbor_fitness
                k = 0
            elif np.random.uniform(0, 1) < acceptance_prob:
                point = neighbor
                current_fitness = neighbor_fitness
                k = 0
            else:
                k += 1

            history.append((point[0], point[1], current_fitness))
            temperature *= self.alpha  # decrease temperature

        return history
