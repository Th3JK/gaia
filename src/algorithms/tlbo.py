import numpy as np
import copy
from src.algorithms.algorithm import Algorithm


class TLBO(Algorithm):
    """
    Teaching-Learning-Based Optimization (TLBO) algorithm.

    Metaheuristic optimization inspired by the teaching–learning process:
      - Teacher phase: individuals learn from the best (teacher).
      - Learner phase: individuals learn from each other.
    """

    def __init__(
        self,
        lower_bound,
        upper_bound,
        function,
        population_size=30,
        dimension=2,
        max_evaluations=3_000,
    ):
        """
        Initialize TLBO algorithm.

        Args:
            lower_bound (float): Lower bound of search space.
            upper_bound (float): Upper bound of search space.
            function (callable): Objective function to minimize.
            population_size (int): Number of individuals in the population.
            dimension (int): Dimensionality of the problem.
            max_evaluations (int): Maximum number of fitness evaluations (OFE).
        """
        super().__init__(lower_bound, upper_bound, function, iterations=max_evaluations)
        self.population_size = population_size
        self.dimension = dimension
        self.max_evaluations = max_evaluations


    @staticmethod
    def generate_population(lower_bound, upper_bound, population_size, dimension):
        """Generate a population uniformly within given bounds."""
        return np.random.uniform(lower_bound, upper_bound, (population_size, dimension))

    @staticmethod
    def evaluate_population(population, function):
        """Evaluate all individuals and return fitness array."""
        return np.array([function(ind) for ind in population])

    def run(self):
        """
        Execute the TLBO algorithm.

        Returns:
            list[list[list[float]]]: Population history for each generation.
        """
        # Initialize population
        population = self.generate_population(self.lower_bound, self.upper_bound, self.population_size, self.dimension)
        fitness = self.evaluate_population(population, self.function)
        evaluations = self.population_size

        # History for visualization/analysis
        history = []

        while evaluations < self.max_evaluations:
            # ----------- Teacher Phase -----------
            best_idx = np.argmin(fitness)
            teacher = population[best_idx]
            mean_pop = np.mean(population, axis=0)
            teaching_factor = np.random.randint(1, 3)  # either 1 or 2

            new_population = population + np.random.rand(self.population_size, self.dimension) * (
                teacher - teaching_factor * mean_pop
            )

            # Bound enforcement
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

            new_fitness = self.evaluate_population(new_population, self.function)
            evaluations += self.population_size

            # Selection
            improved = new_fitness < fitness
            population[improved] = new_population[improved]
            fitness[improved] = new_fitness[improved]

            # ----------- Learner Phase -----------
            for i in range(self.population_size):
                partner_idx = np.random.choice([j for j in range(self.population_size) if j != i])
                partner = population[partner_idx]

                if fitness[partner_idx] < fitness[i]:
                    direction = partner - population[i]
                else:
                    direction = population[i] - partner

                new_individual = population[i] + np.random.rand(self.dimension) * direction
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)

                new_fit = self.function(new_individual)
                evaluations += 1

                if new_fit < fitness[i]:
                    population[i] = new_individual
                    fitness[i] = new_fit

            # Record current population
            history.append(population.tolist())

        return history
