import numpy as np
import random
from src.tsp_utils import TSPUtils


class AntColony:
    """
    Ant Colony Optimization (ACO) algorithm for the Traveling Salesman Problem (TSP).

    Attributes:
        cities (np.ndarray): Coordinates of cities, shape (n_cities, 2).
        n_ants (int): Number of ants in the colony.
        generations (int): Number of iterations (generations).
        evaporation_rate (float): Rate at which pheromone evaporates.
        pheromone_matrix (np.ndarray): Matrix of pheromone levels between cities.
        history (list): List of best paths found per generation.
    """

    def __init__(self, cities, n_ants=20, generations=100, evaporation_rate=0.5):
        """
        Initialize the Ant Colony Optimization algorithm.

        Args:
            cities (np.ndarray): Array of city coordinates, shape (n_cities, 2).
            n_ants (int, optional): Number of ants (default=20).
            generations (int, optional): Number of generations (default=100).
            evaporation_rate (float, optional): Pheromone evaporation rate (default=0.5).
        """
        self.cities = cities
        self.n_ants = n_ants
        self.generations = generations
        self.evaporation_rate = evaporation_rate

        # Precompute matrices
        self.distance_matrix = self._compute_distance_matrix()
        self.pheromone_matrix = np.ones_like(self.distance_matrix)

        # Store history of best paths
        self.history = []

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _compute_distance_matrix(self):
        """Compute symmetric distance matrix between all cities."""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance = TSPUtils.calculate_distance(self.cities[i], self.cities[j])
                matrix[i, j] = matrix[j, i] = distance
        return matrix

    def _compute_visibility(self):
        """Compute inverse of distance matrix (visibility)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            visibility = np.divide(1.0, self.distance_matrix, where=self.distance_matrix != 0)
        np.fill_diagonal(visibility, 0)
        return visibility

    @staticmethod
    def _get_best_path(cities, ants):
        """Return the best path (lowest distance) among all ants."""
        def indices_to_path(idxs):
            return np.array([cities[i] for i in idxs])

        best_path = indices_to_path(ants[0])
        best_distance = TSPUtils.evaluate_individual(best_path)

        for ant in ants[1:]:
            path = indices_to_path(ant)
            distance = TSPUtils.evaluate_individual(path)
            if distance < best_distance:
                best_path, best_distance = path, distance

        return best_path

    def _select_next_city(self, current_city, unvisited, pheromone, visibility):
        """
        Select next city probabilistically based on pheromone and visibility.
        """
        pheromone_values = np.array([pheromone[current_city][c] for c in unvisited])
        visibility_values = np.array([visibility[current_city][c] for c in unvisited])
        weights = pheromone_values * (visibility_values ** 2)

        # Normalize to probabilities
        if np.sum(weights) == 0:
            return random.choice(list(unvisited))
        probabilities = weights / np.sum(weights)
        return random.choices(list(unvisited), weights=probabilities)[0]

    # ------------------------------------------------------------------
    # Core Algorithm
    # ------------------------------------------------------------------

    def run(self, animate_fn=None, animation_path=None):
        """
        Run the Ant Colony Optimization algorithm.

        Args:
            animate_fn (callable, optional): Animation callback.
            animation_path (str, optional): Path to save animation.

        Returns:
            list[np.ndarray]: History of best paths per generation.
        """
        n_cities = len(self.cities)

        for gen in range(self.generations):
            visibility = self._compute_visibility()
            ants = []

            # Each ant builds a path
            for _ in range(self.n_ants):
                start = random.randint(0, n_cities - 1)
                path = [start]
                unvisited = set(range(n_cities)) - {start}

                while unvisited:
                    current = path[-1]
                    next_city = self._select_next_city(current, unvisited, self.pheromone_matrix, visibility)
                    path.append(next_city)
                    unvisited.remove(next_city)

                # Return to start city
                path.append(start)
                ants.append(path)

            # Evaluate and record best path
            best_path = self._get_best_path(self.cities, ants)
            best_distance = TSPUtils.evaluate_individual(best_path)
            self.history.append(best_path)

            # Evaporate pheromones
            self.pheromone_matrix *= (1 - self.evaporation_rate)

            # Update pheromones
            for ant in ants:
                path_distance = TSPUtils.evaluate_individual(np.array([self.cities[i] for i in ant]))
                for i in range(len(ant) - 1):
                    a, b = ant[i], ant[i + 1]
                    delta = 1 / path_distance
                    self.pheromone_matrix[a, b] += delta
                    self.pheromone_matrix[b, a] += delta

            print(
                f"Generation {gen + 1}/{self.generations}, "
                f"Best Path Distance: {best_distance:.2f}"
            )

        # Optionally animate results
        if animate_fn and animation_path:
            animate_fn(self.history, self.cities, "Ant Colony Optimization", animation_path)

        return self.history
