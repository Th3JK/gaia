import numpy as np
import itertools

class Solution:
    """
    A container class that manages the optimization process.

    The Solution class binds together:
      - the search space definition,
      - the target function to minimize,
      - the optimization algorithm, and
      - the history of best points found during execution.

    Attributes:
        dimension (int): Dimensionality of the search space (e.g., 2D for x, y).
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        step (float): Step size for meshgrid generation (used in visualization).
        function (callable): The objective function to minimize.
        algorithm (Algorithm): An instance of the optimization algorithm.
        history (list): List of lists containing the best points found at each step.
    """

    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm):
        """
        Initialize the Solution.

        Args:
            dimension (int): Dimensionality of the search space.
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            step (float): Step size for visualization grid.
            function (callable): The objective function to minimize.
            algorithm (class): The algorithm class (must inherit from Algorithm).
                              An instance is created using the given bounds and function.
        """
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.function = function
        self.algorithm = algorithm(lower_bound, upper_bound, function)
        self.history = []

    def _normalize_point(self, obj):
        """
        Normalize obj into (x, y, z) tuple.
        Accepts sequences or numpy arrays of length 2 or 3.
        If only (x,y) is provided, compute z using self.function.
        Raises ValueError when obj is not a valid point.
        """
        arr = np.asarray(obj, dtype=float)
        if arr.ndim != 1 or arr.size not in (2, 3):
            raise ValueError("Not a point")
        if arr.size == 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
        else:
            x, y = float(arr[0]), float(arr[1])
            z = float(self.function(np.array([x, y])))
            return (x, y, z)

    def _process_population(self, iterator, best_value):
        """
        Process a population (iterator of individuals), append the generation
        to self.history, update and return best_value.
        """
        gen_points = []
        for ind in iterator:
            try:
                p = self._normalize_point(ind)
            except Exception:
                # skip elements that aren't points
                continue
            gen_points.append(p)
            if p[2] < best_value:
                best_value = p[2]

        # store the whole generation so Animator can draw multiple points in a frame
        self.history.append(gen_points)
        return best_value

    def _process_point(self, element, best_value):
        """
        Process a single point element (x,y) or (x,y,z). If it improves best_value,
        record it in history (as a single-item list) and return the updated best_value.
        """
        try:
            p = self._normalize_point(element)
        except Exception:
            return best_value

        if p[2] < best_value:
            best_value = p[2]
            # keep same shape as before: a list containing the point
            self.history.append([p])

        return best_value

    def _is_population(self, element):
        """
        Determine whether `element` is a population (an iterable of points).
        Returns (is_population: bool, iterator_or_none).

        - Avoid treating strings/bytes as populations.
        - Uses itertools.tee so the returned iterator can be consumed safely
          (generators won't lose their first item).
        """
        if isinstance(element, (str, bytes)):
            return False, None

        try:
            iterator = iter(element)
        except TypeError:
            return False, None

        # Duplicate iterator so we can peek first element safely.
        it1, it2 = itertools.tee(iterator)
        try:
            first = next(it1)
        except StopIteration:
            # empty iterable — not considered a population here
            return False, None

        # If the first member can be normalized to a point, treat `element` as a population.
        try:
            self._normalize_point(first)
            return True, it2
        except Exception:
            return False, None

    def find_minimum(self):
        """
        Run the optimization algorithm to find the minimum of the function.

        Supports algorithm.run() outputs that are either:
          - a flat iterable of points: (x, y) or (x, y, z), OR
          - an iterable of populations (generations), where each population is an
            iterable of individuals (each individual is (x, y) or (x, y, z)).

        Returns:
            float: The best (minimum) function value found.
        """
        best_value = float("inf")
        self.history = []

        all_points = self.algorithm.run()

        for element in all_points:
            is_pop, iterator = self._is_population(element)
            if is_pop:
                best_value = self._process_population(iterator, best_value)
            else:
                best_value = self._process_point(element, best_value)

        return best_value
