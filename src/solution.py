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

    def find_minimum(self):
        """
        Run the optimization algorithm to find the minimum of the function.

        Tracks improvements in the function value and records them in `history`.

        Returns:
            float: The best (minimum) function value found.
        """
        best_value = float("inf")
        self.history = []

        # Run the algorithm and collect all visited points
        all_points = self.algorithm.run()

        # Track improvements
        for point in all_points:
            x, y, z = point
            if z < best_value:
                best_value = z
                self.history.append([point])  # Store the improvement

        return best_value
