import numpy as np

class Function:
    """
    A collection of commonly used benchmark functions for optimization.
    """

    @staticmethod
    def sphere(params: np.ndarray = np.arange(-5.12, 5.12, 0.1)) -> float:
        """
        Sphere function.

        f(x) = sum(x_i^2)

        Global minimum at x = 0, f(x) = 0.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        return float(np.sum(params**2))

    @staticmethod
    def ackley(params: np.ndarray = np.arange(-32.768, 32.768, 1.0), a = 20, b = .2, c = 2 * np.pi):
        """
        Ackley function.

        f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + e

        Global minimum at x = 0, f(x) = 0.

        Parameters:
            params (np.ndarray): Input vector.
            a (float, optional): Default 20.
            b (float, optional): Default 0.2.
            c (float, optional): Default 2*pi.

        Returns:
            float: Function value.
        """
        n = len(params)

        sum1 = np.sum(params**2)
        sum2 = np.sum(np.cos(c * params))

        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)

        return term1 + term2 + a + np.e

    @staticmethod
    def rastrigin(params: np.ndarray = np.arange(-5.12, 5.12, 0.6)) -> float:
        """
        Rastrigin function.

        f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))

        Global minimum at x = 0, f(x) = 0.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        n = len(params)
        total = np.sum(params**2 - 10 * np.cos(2 * np.pi * params))

        return 10 * n + total

    @staticmethod
    def rosenbrock(params: np.ndarray = np.arange(-10, 10, 0.3)) -> float:
        """
        Rosenbrock function.

        f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

        Global minimum at x = 1, f(x) = 0.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        first = params[:-1]
        second = params[1:]
        total = np.sum(100 * (second - first**2)**2 + (first - 1)**2)

        return total

    @staticmethod
    def rosenbrock_modified(params: np.ndarray) -> float:
        """
        Modified Rosenbrock function with scaling and normalization.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Normalized function value.
        """
        scaled = 15 * params - 5
        first = scaled[0:3]
        second = scaled[1:4]
        total = np.sum(100 * (second - first**2)**2 + (1 - first)**2)

        return (total - 3.827e5) / 3.755e5

    @staticmethod
    def griewank(params: np.ndarray = np.arange(-50, 50, 1.0)) -> float:
        """
        Griewank function.

        f(x) = sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i))) + 1

        Global minimum at x = 0, f(x) = 0.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        n = len(params)
        indices = np.arange(1, n + 1)
        
        term1 = np.sum(params**2 / 4000)
        term2 = np.prod(np.cos(params / np.sqrt(indices)))

        return term1 - term2 + 1

    @staticmethod
    def schwefel(params: np.ndarray = np.arange(-500, 500, 2.5)) -> float:
        """
        Schwefel function.

        f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))

        Global minimum at x_i = 420.9687, f(x) ≈ 0.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        n = len(params)
        term = np.sum(params * np.sin(np.sqrt(np.abs(params))))

        return 418.9829 * n - term

    @staticmethod
    def levy(params: np.ndarray = np.arange(-10, 10, 0.1)) -> float:
        """
        Levy function.

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        w = 1 + (params - 1) / 4
        first = w[:-1]
        second = w[1:]

        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((first -1)**2 * (1 + 10 * np.sin(np.pi * second)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        return term1 + term2 + term3

    @staticmethod
    def michalewicz(params: np.ndarray = np.arange(0, np.pi, 0.1), constant = 10) -> float:
        """
        Michalewicz function.

        f(x) = -sum(sin(x_i) * (sin(i * x_i^2 / pi))^(2*m))

        Parameters:
            params (np.ndarray): Input vector.
            constant (int, optional): Shape parameter m. Default is 10.

        Returns:
            float: Function value.
        """
        n = len(params)
        indices = np.arange(1, n + 1)
        term = np.sum(np.sin(params) * (np.sin(indices * params**2 / np.pi))**(2*constant))
        return -term

    @staticmethod
    def zakharov(params: np.ndarray = np.arange(-10, 10, 0.4)) -> float:
        """
        Zakharov function.

        f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4

        Parameters:
            params (np.ndarray): Input vector.

        Returns:
            float: Function value.
        """
        n = len(params)
        indices = np.arange(1, n + 1)

        term1 = np.sum(params**2)
        term2 = np.sum(0.5 * indices * params)

        return term1 + term2**2 + term2**4
    
    @staticmethod
    def get_all_functions() -> list:
        """
        This method returns all test functions that have declared default values.
        """
        return [
            (Function.sphere, (-5.12, 5.12, 0.1)),
            (Function.ackley, (-32.768, 32.768, 1.0)),
            (Function.rastrigin, (-5.12, 5.12, 0.6)),
            (Function.rosenbrock, (-10, 10, 0.3)),
            (Function.griewank, (-10, 10, 0.5)),
            (Function.schwefel, (-500, 500, 2.5)),
            (Function.levy, (-10, 10, 0.1)),
            (Function.michalewicz, (0, np.pi, 0.1)),
            (Function.zakharov, (-10, 10, 0.4))
        ]