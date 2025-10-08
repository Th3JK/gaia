import numpy as np

class TSPUtils:
    """
    Utility functions for the Traveling Salesman Problem (TSP).
    Provides methods for distance calculation, random generation,
    and population management using only NumPy operations.
    """

    @staticmethod
    def calculate_distance(city1, city2):
        """
        Calculate the Euclidean distance between two cities.

        Args:
            city1 (np.ndarray): Coordinates [x, y] of the first city.
            city2 (np.ndarray): Coordinates [x, y] of the second city.

        Returns:
            float: Euclidean distance between city1 and city2.
        """
        return np.linalg.norm(city1 - city2)

    @staticmethod
    def evaluate_individual(individual):
        """
        Compute the total path distance for a TSP individual (path).

        Args:
            individual (np.ndarray): Array of shape (n_cities, 2)
                                     representing the ordered path.

        Returns:
            float: Total path distance.
        """
        diffs = individual[1:] - individual[:-1]
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    @staticmethod
    def generate_cities(n, x_range=(0, 100), y_range=(0, 100)):
        """
        Generate random city coordinates within given bounds.

        Args:
            n (int): Number of cities to generate.
            x_range (tuple): (min, max) range for x coordinates.
            y_range (tuple): (min, max) range for y coordinates.

        Returns:
            np.ndarray: Array of shape (n, 2) containing city coordinates.
        """
        xs = np.random.uniform(*x_range, n)
        ys = np.random.uniform(*y_range, n)
        return np.stack((xs, ys), axis=1)

    @staticmethod
    def generate_individual(cities):
        """
        Create a random TSP path by shuffling the city order.

        Args:
            cities (np.ndarray): Array of shape (n, 2) representing city coordinates.

        Returns:
            np.ndarray: Shuffled copy of `cities` representing a new individual.
        """
        return cities[np.random.permutation(len(cities))]

    @staticmethod
    def generate_population(cities, size):
        """
        Generate an initial random population of individuals.

        Args:
            cities (np.ndarray): Array of shape (n, 2) representing city coordinates.
            size (int): Population size.

        Returns:
            np.ndarray: Array of shape (size, n, 2) representing all individuals.
        """
        n = len(cities)
        population = np.empty((size, n, 2))
        for i in range(size):
            population[i] = TSPUtils.generate_individual(cities)
        return population
