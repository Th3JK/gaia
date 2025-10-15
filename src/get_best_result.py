import numpy as np

def get_best_result(results, test_function):
    """Return the best (minimum) fitness from the last generation of results."""
    last_population = results[-1]
    fitness_values = [test_function(np.array(ind)) for ind in last_population]
    return float(np.min(fitness_values))
