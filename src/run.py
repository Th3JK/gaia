import matplotlib

from function import Function
from solution import Solution
from animator import Animator
from algorithms.blind_search import BlindSearch

def run(algorithm):
    matplotlib.use('TkAgg')

    for function, params in Function.get_all_functions():
        solution = Solution(2, params[0], params[1], params[2], function, algorithm)
        best = solution.find_minimum()
        print(f"Function: {function.__name__}, Algorithm: {algorithm.__name__}, Best found solution: {best}")
        animator = Animator(solution)
        animator.save()

run(BlindSearch)