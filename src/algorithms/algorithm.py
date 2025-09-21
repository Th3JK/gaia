class Algorithm: 
    @abstractmethod
    def __init__(self, lower_bound, upper_bound, function, iterations=1000):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.function = function
        self.iterations = iterations

    @abstractmethod
    def run(self):
        pass