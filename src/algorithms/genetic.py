import numpy as np
from src.tsp_utils import TSPUtils


class Genetic:
    """
    Genetic Algorithm for solving the Traveling Salesman Problem (TSP).

    Attributes:
        cities (np.ndarray): Array of city coordinates of shape (n_cities, 2).
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to evolve.
        mutation_prob (float): Probability of mutation.
    """

    def __init__(self, cities, population_size=20, generations=200, mutation_prob=0.2):
        """
        Initialize the Genetic Algorithm for TSP.

        Args:
            cities (np.ndarray): City coordinates.
            population_size (int, optional): Number of individuals (default is 20).
            generations (int, optional): Number of generations (default is 200).
            mutation_prob (float, optional): Probability of mutation (default is 0.2).
        """
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.population = TSPUtils.generate_population(cities, population_size)
        self.history = []

    @staticmethod
    def order_crossover(parent_a, parent_b):
        """
        Perform Order Crossover (OX) between two parents.

        Args:
            parent_a (np.ndarray): First parent, shape (n_cities, 2).
            parent_b (np.ndarray): Second parent, shape (n_cities, 2).

        Returns:
            np.ndarray: Offspring individual, shape (n_cities, 2).
        """
        size = parent_a.shape[0]
        start, end = np.sort(np.random.choice(size, 2, replace=False))

        offspring = np.empty_like(parent_a)
        offspring[:] = np.nan  # placeholder for missing genes
        offspring[start:end] = parent_a[start:end]

        # mask to avoid duplicates
        mask = np.isin(parent_b.view([('', parent_b.dtype)] * 2),
                       offspring.view([('', offspring.dtype)] * 2),
                       invert=True).reshape(size)

        fill_values = parent_b[mask]
        fill_positions = np.arange(size)[np.isnan(offspring[:, 0])]
        offspring[fill_positions] = fill_values

        return offspring

    @staticmethod
    def mutate(individual):
        """
        Mutate an individual by swapping two random cities.

        Args:
            individual (np.ndarray): TSP path of shape (n_cities, 2).

        Returns:
            np.ndarray: Mutated individual.
        """
        mutated = np.copy(individual)
        i, j = np.random.choice(mutated.shape[0], 2, replace=False)
        mutated[[i, j]] = mutated[[j, i]]
        return mutated

    def get_best_individual(self):
        """
        Retrieve the best individual in the current population.

        Returns:
            np.ndarray: Best path found (shape (n_cities, 2)).
        """
        distances = np.apply_along_axis(
            lambda idx: TSPUtils.evaluate_individual(self.population[idx]),
            0,
            np.arange(self.population.shape[0])
        )
        best_index = np.argmin(distances)
        return self.population[best_index]

    def run(self):
        """
        Execute the Genetic Algorithm to solve the TSP.

        Returns:
            list of np.ndarray: History of best individuals per generation.
        """
        for gen in range(self.generations):
            new_population = np.copy(self.population)

            for i in range(self.population_size):
                # Select two distinct parents
                idxs = np.random.choice(self.population_size, 2, replace=False)
                parent_a = self.population[idxs[0]]
                parent_b = self.population[idxs[1]]

                offspring = self.order_crossover(parent_a, parent_b)

                # Mutation
                if np.random.rand() < self.mutation_prob:
                    offspring = self.mutate(offspring)

                # Replace if better
                if TSPUtils.evaluate_individual(offspring) < TSPUtils.evaluate_individual(parent_a):
                    new_population[i] = offspring

            self.population = new_population
            best = self.get_best_individual()
            self.history.append(best)

            print(
                f"Generation {gen + 1}/{self.generations}, "
                f"Best Path Distance: {TSPUtils.evaluate_individual(best):.2f}"
            )

        return self.history
