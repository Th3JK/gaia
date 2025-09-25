from abc import abstractmethod

class Algorithm:
    """
    Abstract base class for optimization algorithms.

    This class defines a common interface for optimization algorithms
    by enforcing initialization with bounds, a target function, and
    a maximum number of iterations, as well as a `run` method that
    must be implemented by subclasses.
    """

    @abstractmethod
    def __init__(self, lower_bound, upper_bound, function, iterations=1000):
        """
        Initialize the optimization algorithm.

        Args:
            lower_bound (float): Lower bound of the search space.
            upper_bound (float): Upper bound of the search space.
            function (callable): The objective function to optimize.
                                 Must accept a NumPy array as input.
            iterations (int, optional): Maximum number of iterations
                                        allowed during the optimization.
                                        Defaults to 1000.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.function = function
        self.iterations = iterations

    @abstractmethod
    def run(self):
        """
        Run the optimization algorithm.

        This method must be implemented by subclasses to define
        how the algorithm explores the search space and returns
        the optimization results.

        Returns:
            Any: The output of the optimization algorithm.
                 For example, it could return the best solution found,
                 a history of solutions, or other relevant data.
        """
        pass
