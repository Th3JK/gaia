import numpy as np
import random
from src.algorithms.algorithm import Algorithm


class Firefly(Algorithm):
    """
    Firefly Algorithm optimization method.

    This metaheuristic algorithm is inspired by the flashing behavior of fireflies.
    Each firefly represents a potential solution, and they move toward brighter
    (better) fireflies according to their attractiveness and a randomization factor.
    """

    def __init__(
        self,
        lower_bound,
        upper_bound,
        function,
        population_size=20,
        generations=200,
        alpha=0.3,
        beta_zero=1.0,
        dimension=2
    ):
        """
        Initialize the Firefly Algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): Function to minimize. Must accept a NumPy array.
            population_size (int, optional): Number of fireflies in the population.
                                             Defaults to 20.
            generations (int, optional): Number of generations (iterations) to run.
                                         Defaults to 200.
            alpha (float, optional): Randomization parameter controlling randomness.
                                     Defaults to 0.3.
            beta_zero (float, optional): Base attractiveness between fireflies.
                                         Defaults to 1.0.
            dimension (int, optional): Dimensionality of the search space.
                                       Defaults to 2.
        """
        super().__init__(lower_bound, upper_bound, function, iterations=generations)
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta_zero = beta_zero
        self.dimension = dimension

    @staticmethod
    def _distance(x1, x2):
        """Compute the Euclidean distance between two points."""
        return np.linalg.norm(x1 - x2)

    def _generate_population(self):
        """Generate an initial population of fireflies within bounds."""
        return [
            np.array([random.uniform(self.lower_bound, self.upper_bound)
                      for _ in range(self.dimension)])
            for _ in range(self.population_size)
        ]

    def _get_light_intensities(self, population):
        """Compute the light intensities (fitness values) for each firefly."""
        return [self.function(x) for x in population]

    def _random_vector(self):
        """Generate a normally distributed random vector."""
        return np.random.normal(0, 1, self.dimension)

    def run(self):
        """
        Run the Firefly Algorithm optimization process.

        Returns:
            list of list of lists: A history of all firefly positions across generations.
                                   Each entry represents one generation and contains a list
                                   of firefly positions as [x1, x2, ...].
        """
        # Initialize population and compute their light intensities
        population = self._generate_population()
        intensities = self._get_light_intensities(population)

        # Store all generations
        history = []

        for _ in range(self.generations):
            generation_positions = []

            for i in range(self.population_size):
                for j in range(self.population_size):
                    # Firefly i moves toward firefly j if j is brighter (lower fitness)
                    if intensities[i] > intensities[j]:
                        beta = self.beta_zero / (1 + self._distance(population[i], population[j]))
                        population[i] += beta * (population[j] - population[i]) + self.alpha * self._random_vector()

                # Recalculate intensity for updated position
                intensities[i] = self.function(population[i])
                generation_positions.append(list(population[i]))

            # Record generation
            history.append(generation_positions)

        return history