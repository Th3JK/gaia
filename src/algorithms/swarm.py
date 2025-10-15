import numpy as np

from src.algorithms.algorithm import Algorithm


class Swarm(Algorithm):

    def __init__(
            self,
            lower_bound,
            upper_bound,
            function,
            max_iterations=10_000,
            population_size=20,
            max_cycles=500,
            cognitive_coeff=2,
            social_coeff=2,
            velocity_max=1,
            velocity_min=-1,
            inertia_max=0.9,
            inertia_min=0.4,
            dimension=2,
        ):
        """
        Particle Swarm Optimization (PSO) implementation.

        Args:
            lower_bound (float): Lower bound for search space.
            upper_bound (float): Upper bound for search space.
            function (callable): Objective function to minimize.
            max_iterations (int): Number of iterations for Algorithm base class.
            population_size (int): Number of particles in the swarm.
            max_cycles (int): Number of iterations (generations) for PSO.
            cognitive_coeff (float): Cognitive acceleration coefficient (C1).
            social_coeff (float): Social acceleration coefficient (C2).
            velocity_max (float): Maximum velocity limit.
            velocity_min (float): Minimum velocity limit.
            inertia_max (float): Maximum inertia weight.
            inertia_min (float): Minimum inertia weight.
            dimension (int): Dimensionality of the search space.
        """
        super().__init__(lower_bound, upper_bound, function, max_iterations)

        # Rename to clearer attributes
        self.population_size = population_size
        self.cognitive_coeff = cognitive_coeff  # c1
        self.social_coeff = social_coeff        # c2
        self.max_cycles = max_cycles            # m_max
        self.velocity_max = velocity_max
        self.velocity_min = velocity_min
        self.inertia_max = inertia_max          # w_max
        self.inertia_min = inertia_min          # w_min
        self.dimension = dimension


    @staticmethod
    def generate_population(lower_bound, upper_bound, num_particles, dimension=2):
        """Generate a population of particles uniformly within the given bounds."""
        return [np.random.uniform(lower_bound, upper_bound, dimension).tolist() for _ in range(num_particles)]

    @staticmethod
    def update_velocity(current_velocity, position, personal_best, global_best,
                        inertia_weight, cognitive_coeff, social_coeff,
                        velocity_min, velocity_max):
        """Update velocity of a particle according to the PSO formula."""
        r1, r2 = np.random.rand(), np.random.rand()
        new_velocity = (
            inertia_weight * current_velocity
            + cognitive_coeff * r1 * (personal_best - position)
            + social_coeff * r2 * (global_best - position)
        )
        return np.clip(new_velocity, velocity_min, velocity_max)

    @staticmethod
    def update_position(position, velocity, lower_bound, upper_bound):
        """Update the position of a particle and enforce bounds."""
        new_position = position + velocity
        return np.clip(new_position, lower_bound, upper_bound)

    @staticmethod
    def get_best_position(population, objective_function):
        """Find the best particle position in the population."""
        best = population[0]
        best_value = objective_function(best)
        for particle in population:
            value = objective_function(particle)
            if value < best_value:
                best_value = value
                best = particle
        return np.array(best)

    def run(self):
        """
        Run the Particle Swarm Optimization (PSO) algorithm.

        Returns:
            list: A list of all particle positions for each iteration (for visualization or analysis).
        """
        # Initialize swarm
        swarm = np.array(
            self.generate_population(self.lower_bound, self.upper_bound, self.population_size, self.dimension)
        )
        personal_best_positions = np.copy(swarm)
        global_best_position = self.get_best_position(swarm, self.function)

        # Initialize velocities
        velocities = np.array(
            self.generate_population(self.velocity_min, self.velocity_max, self.population_size, self.dimension)
        )

        # Track history for visualization/analysis
        all_generations = []

        for cycle in range(self.max_cycles):
            iteration_positions = []
            # Decrease inertia weight linearly
            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (cycle / self.max_cycles)

            for i in range(self.population_size):
                position = swarm[i]
                personal_best = personal_best_positions[i]

                # Update velocity and position
                velocities[i] = self.update_velocity(
                    velocities[i], position, personal_best, global_best_position,
                    inertia_weight, self.cognitive_coeff, self.social_coeff,
                    self.velocity_min, self.velocity_max
                )
                swarm[i] = self.update_position(position, velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness
                current_value = self.function(swarm[i])
                personal_best_value = self.function(personal_best)

                # Update personal best
                if current_value < personal_best_value:
                    personal_best_positions[i] = swarm[i]
                    # Update global best
                    if self.function(personal_best_positions[i]) < self.function(global_best_position):
                        global_best_position = personal_best_positions[i]

                iteration_positions.append(swarm[i].tolist())

            all_generations.append(iteration_positions)

        return all_generations