import matplotlib

from src.function import Function
from src.solution import Solution
from src.animator import Animator


def benchmark_algorithm(algorithm):
    """
    Benchmark a given optimization algorithm on a suite of test functions.

    For each function defined in `Function.get_all_functions()`:
      - Create a `Solution` object with the algorithm and function.
      - Run the optimization to find the best minimum.
      - Print the best solution found.
      - Generate and save an animation of the optimization process.

    Args:
        algorithm (class): The optimization algorithm class to benchmark.
                           Must inherit from `Algorithm`.

    Output:
        Console:
            Prints the best result for each function in the form:
            "Function: <function_name>, Algorithm: <algorithm_name>, Best found solution: <value>"

        Files:
            Saves a GIF animation for each function in:
            `../assets/{algorithm_name}/{function_name}.gif`
    """
    matplotlib.use("TkAgg")

    for function, params in Function.get_all_functions():
        solution = Solution(2, params[0], params[1], params[2], function, algorithm)
        best = solution.find_minimum()
        print(
            f"Function: {function.__name__}, Algorithm: {algorithm.__name__}, Best found solution: {best}"
        )

        animator = Animator(solution)
        animator.save()
