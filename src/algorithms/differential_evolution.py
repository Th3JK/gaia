import numpy as np
import copy

from src.algorithms.algorithm import Algorithm


class DifferentialEvolution(Algorithm):
    def __init__(self, lower_bound, upper_bound, function, iterations=10_000, individuals=10, mutation=.5, crossover=.5, cycles=100):
        super().__init__(lower_bound, upper_bound, function, iterations)
        self.individuals = individuals
        self.mutation = mutation
        self.crossover = crossover
        self.cycles = cycles

    @staticmethod
    def generate_population(lower_bound, upper_bound, input_np, dimension=2):
        """Generate a population of NP individuals with the given dimension."""
        return [np.random.uniform(lower_bound, upper_bound, dimension).tolist() for _ in range(input_np)]

    @staticmethod
    def get_random_parents(population, exclude):
        """Get random parents for mutation."""
        result = [i for i in range(len(population)) if population[i] not in exclude]
        return np.random.choice(result)

    def run(self):
        """
        Run differential evolution.

        Returns:
            list: history of population states (one list-of-individuals per generation)
        """
        # History of populations
        history = []

        # Initialize population
        pop = self.generate_population(self.lower_bound, self.upper_bound, self.individuals, dimension=2)

        for g in range(self.cycles):
            new_population = copy.deepcopy(pop)

            for i, individual in enumerate(pop):
                # pick three distinct parents (indices) not equal to the target individual
                # ensure ints because np.random.choice may return numpy scalar
                r1_i = int(self.get_random_parents(pop, [individual]))
                r2_i = int(self.get_random_parents(pop, [individual, pop[r1_i]]))
                r3_i = int(self.get_random_parents(pop, [individual, pop[r1_i], pop[r2_i]]))

                # fetch parent vectors from the new_population (as in classical DE)
                r1 = np.array(new_population[r1_i])
                r2 = np.array(new_population[r2_i])
                r3 = np.array(new_population[r3_i])

                # mutation
                mutated = (r1 - r2) * self.mutation + r3

                # crossover -> trial vector
                trial = np.zeros(len(individual))
                j_rnd = np.random.randint(0, len(individual))  # ensure at least one gene comes from mutated

                for j in range(len(individual)):
                    if np.random.uniform() < self.crossover or j == j_rnd:
                        trial[j] = mutated[j]
                    else:
                        trial[j] = individual[j]

                # enforce bounds
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # selection: minimize the provided function
                if self.function(np.array(trial)) <= self.function(np.array(individual)):
                    new_population[i] = list(trial)

            # store generation and move to next
            history.append(copy.deepcopy(new_population))
            pop = new_population

        return history