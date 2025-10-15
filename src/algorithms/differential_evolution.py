import numpy as np
import copy

from src.algorithms.algorithm import Algorithm


class DifferentialEvolution(Algorithm):
    """
    Differential Evolution optimization algorithm.

    This population-based stochastic optimization method evolves a population
    of candidate solutions using operations inspired by natural evolution:
    mutation, crossover, and selection.

    Each generation produces new trial solutions by combining existing ones,
    and individuals are replaced by better-performing offspring based on
    the provided objective function.

    The algorithm is suitable for continuous, nonlinear, and non-differentiable
    optimization problems.
    """

    def __init__(self, lower_bound, upper_bound, function, iterations=10_000,
                 individuals=10, mutation=.5, crossover=.5, cycles=100):
        """
        Initialize the DifferentialEvolution algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): Objective function to minimize.
                                 Must accept a NumPy array and return a scalar value.
            iterations (int, optional): Total number of function evaluations allowed (default is 10,000).
            individuals (int, optional): Number of individuals (population size) (default is 10).
            mutation (float, optional): Mutation factor (scales the differential variation) (default is 0.5).
            crossover (float, optional): Probability of crossover between individuals (default is 0.5).
            cycles (int, optional): Number of generations (iterations of evolution) (default is 100).
        """
        super().__init__(lower_bound, upper_bound, function, iterations)
        self.individuals = individuals
        self.mutation = mutation
        self.crossover = crossover
        self.cycles = cycles

    @staticmethod
    def generate_population(lower_bound, upper_bound, input_np, dimension=2):
        """
        Generate an initial population of individuals.

        Each individual is a vector sampled uniformly within the search bounds.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            input_np (int): Number of individuals in the population.
            dimension (int, optional): Dimensionality of each individual (default is 2).

        Returns:
            list of list[float]: A list containing the initial population,
                                 where each individual is represented as a list of floats.
        """
        return [np.random.uniform(lower_bound, upper_bound, dimension).tolist()
                for _ in range(input_np)]

    @staticmethod
    def get_random_parents(population, exclude):
        """
        Select a random individual index from the population,
        excluding specific individuals.

        Args:
            population (list): Current list of individuals.
            exclude (list): List of individuals to exclude from selection.

        Returns:
            int: Index of a randomly chosen individual not in the exclude list.
        """
        result = [i for i in range(len(population)) if population[i] not in exclude]
        return np.random.choice(result)

    def run(self):
        """
        Run the Differential Evolution optimization process.

        This method executes the full evolutionary loop:
        - Initializes a population of candidate solutions.
        - Iteratively applies mutation, crossover, and selection.
        - Retains better-performing individuals over generations.

        Returns:
            list of list[list[float]]: History of population states over generations.
                                       Each element represents a generation, which is
                                       a list of individuals (each being a list of coordinates).
        """
        # History of all generations
        history = []

        # Initialize population
        pop = self.generate_population(self.lower_bound, self.upper_bound,
                                       self.individuals, dimension=2)

        for g in range(self.cycles):
            new_population = copy.deepcopy(pop)

            for i, individual in enumerate(pop):
                # Select three distinct random parents different from the current individual
                r1_i = int(self.get_random_parents(pop, [individual]))
                r2_i = int(self.get_random_parents(pop, [individual, pop[r1_i]]))
                r3_i = int(self.get_random_parents(pop, [individual, pop[r1_i], pop[r2_i]]))

                # Fetch parent vectors
                r1 = np.array(new_population[r1_i])
                r2 = np.array(new_population[r2_i])
                r3 = np.array(new_population[r3_i])

                # Mutation: create a mutant vector
                mutated = (r1 - r2) * self.mutation + r3

                # Crossover: create a trial vector
                trial = np.zeros(len(individual))
                j_rnd = np.random.randint(0, len(individual))  # at least one gene from mutant

                for j in range(len(individual)):
                    if np.random.uniform() < self.crossover or j == j_rnd:
                        trial[j] = mutated[j]
                    else:
                        trial[j] = individual[j]

                # Enforce bounds
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection: accept the trial if it performs better
                if self.function(np.array(trial)) <= self.function(np.array(individual)):
                    new_population[i] = list(trial)

            # Record generation
            history.append(copy.deepcopy(new_population))
            pop = new_population

        return history
