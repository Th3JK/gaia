import numpy as np
import copy

from src.algorithms.algorithm import Algorithm


class Soma(Algorithm):
    def __init__(
        self,
        lower_bound,
        upper_bound,
        function,
        max_iterations=10_000,
        population_size=20,
        migrations=100,
        step_size=0.11,
        prt_probability=0.4,
        path_length=3.0,
        dimension=2,
    ):
        """
        Self-Organizing Migration Algorithm (SOMA) - all-to-one variant.

        Args:
            lower_bound (float): Lower bound for search space (scalar or array-like).
            upper_bound (float): Upper bound for search space (scalar or array-like).
            function (callable): Objective function to minimize. Accepts numpy array.
            max_iterations (int): For base Algorithm class compatibility (not used directly here).
            population_size (int): Number of individuals in the population.
            migrations (int): Number of migration cycles (generations).
            step_size (float): Step increment along migration path.
            prt_probability (float): Probability mask for which genes move during a step.
            path_length (float): Maximum length of the migration path (multiplier).
            dimension (int): Dimensionality of search space.
        """
        super().__init__(lower_bound, upper_bound, function, max_iterations)

        # Clear attribute names
        self.population_size = population_size
        self.migrations = migrations
        self.step_size = step_size
        self.prt_probability = prt_probability
        self.path_length = path_length
        self.dimension = dimension

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def generate_population(lower_bound, upper_bound, num_individuals, dimension=2):
        """Generate a population of individuals uniformly within the given bounds."""
        return [np.random.uniform(lower_bound, upper_bound, dimension).tolist() for _ in range(num_individuals)]

    @staticmethod
    def evaluate_population(population, objective_function):
        """Return (best_individual, best_value) in `population` for a minimization problem."""
        best_ind = population[0]
        best_val = objective_function(np.array(best_ind))
        for ind in population:
            val = objective_function(np.array(ind))
            if val < best_val:
                best_val = val
                best_ind = ind
        return np.array(best_ind), best_val

    @staticmethod
    def migrate_individual(individual, best_individual, objective_function,
                           lower_bound, upper_bound, step_size, path_length, prt_probability):
        """
        Migrate a single individual towards the best_individual.
        Uses PRT vector to probabilistically select components to move.
        """
        ind = np.array(individual, dtype=float)
        best = np.array(best_individual, dtype=float)

        # iterate along the path but do not include the final (path_length) if using same convention
        for s in np.arange(0, path_length, step_size):
            prt_vector = (np.random.rand(len(ind)) < prt_probability).astype(float)
            new_pos = ind + s * (best - ind) * prt_vector
            # enforce bounds
            new_pos = np.clip(new_pos, lower_bound, upper_bound)

            if objective_function(new_pos) < objective_function(ind):
                ind = new_pos

        return ind

    # ----------------------------
    # Main algorithm
    # ----------------------------
    def run(self):
        """
        Run SOMA (all-to-one).

        Returns:
            list: history of populations (list of lists; one list-per-migration containing individuals as lists)
        """
        # initialize population and best
        pop = np.array(self.generate_population(self.lower_bound, self.upper_bound, self.population_size, self.dimension))
        best_ind, _ = self.evaluate_population(pop, self.function)

        history = []

        for mig in range(self.migrations):
            new_pop = []

            for ind in pop:
                migrated = self.migrate_individual(
                    ind,
                    best_ind,
                    self.function,
                    self.lower_bound,
                    self.upper_bound,
                    self.step_size,
                    self.path_length,
                    self.prt_probability,
                )
                new_pop.append(migrated)

            # update population and best
            pop = np.array(new_pop)
            best_ind, _ = self.evaluate_population(pop, self.function)

            # record generation (convert np arrays to lists for serializability/consistency)
            history.append([p.tolist() for p in pop])

        return history
