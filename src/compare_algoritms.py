import os
from src.algorithms.differential_evolution import DifferentialEvolution
from src.algorithms.swarm import Swarm
from src.algorithms.soma import Soma
from src.algorithms.firefly import Firefly
from src.algorithms.tlbo import TLBO
from src.function import Function
from src.solution import Solution

def run_algorithm(Algorithm, lower_bound, upper_bound, func, **kwargs):
    solver = Solution(
        dimension=kwargs.get("dimension", 2),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        step=0.1,
        function=func,
        algorithm=Algorithm,
        **kwargs,
    )
    return solver.find_minimum()

def compare_algorithms(num_experiments=10, save_dir="../results"):
    """
    Run DE, PSO, SOMA, FA, TLBO on benchmark functions and save results to CSV.
    """
    benchmark_functions = {
        "rastrigin": Function.rastrigin,
        "ackley": Function.ackley,
        "sphere": Function.sphere,
        "rosenbrock": Function.rosenbrock,
    }

    results = {}

    for i in range(1, num_experiments + 1):
        print(f"\nExperiment {i}/{num_experiments}")

        for name, func in benchmark_functions.items():
            lower_bound, upper_bound = -5.12, 5.12
            print(f"→ Running {name}")

            de_result = run_algorithm(DifferentialEvolution, lower_bound, upper_bound, func)
            pso_result = run_algorithm(Swarm, lower_bound, upper_bound, func)
            soma_result = run_algorithm(Soma, lower_bound, upper_bound, func)
            fa_result = run_algorithm(Firefly, lower_bound, upper_bound, func)
            tlbo_result = run_algorithm(TLBO, lower_bound, upper_bound, func)

            # Store results for this function
            results.setdefault(name, []).append([
                i,
                de_result,
                pso_result,
                soma_result,
                fa_result,
                tlbo_result,
            ])

    # Save CSV
    os.makedirs(save_dir, exist_ok=True)
    save_results_to_csv(results, os.path.join(save_dir, "comparison_results.csv"))
