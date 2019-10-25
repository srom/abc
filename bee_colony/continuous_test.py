import unittest

import numpy as np

from .continuous import ABC
from .utils import uniform_init


class TestArtificialBeeColony(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)
        np.random.seed(123)

    def test_univariate(self):

        def fitness_fn(x):
            """
            f(x) = x^2
            """
            return x[:,0]**2

        bounds = [[-5, 5]]

        abc = ABC(
            population_size=10,
            fitness_fn=fitness_fn,
            init_fn=uniform_init(bounds)
        )
        best_solution, best_fitness = abc.search(50)

        # Solution array should be one dimensional
        self.assertEqual(1, len(best_solution))

        # Minimum is close to x* = 0
        self.assertTrue(np.isclose(best_solution[0], 0., rtol=1e-4, atol=1e-4))

        # Minimum value is close to f(x*) = 0
        self.assertTrue(np.isclose(best_fitness, 0., rtol=1e-4, atol=1e-4))

    def test_six_hump_camel_fn(self):

        def fitness_fn(x):
            """
            Six-Hump Camel Function
            https://www.sfu.ca/~ssurjano/camel6.html
            """
            return (
                (4 - 2.1 * x[:,0]**2 + x[:,0]**4 / 3) * x[:,0]**2 +
                x[:,0] * x[:,1] +
                (-4 + 4 * x[:,1]**2) * x[:,1]**2
            )

        bounds = [[-3, 3], [-2, 2]]

        abc = ABC(
            population_size=10,
            fitness_fn=fitness_fn,
            init_fn=uniform_init(bounds)
        )
        (x1, x2), best_fitness = abc.search(100)

        # Assert global minimum has been reached
        cond = (
            (
                np.isclose(x1, 0.0898, rtol=1e-3) and
                np.isclose(x2, -0.7126, rtol=1e-3)
            ) or
            (
                np.isclose(x1, -0.0898, rtol=1e-3) and
                np.isclose(x2, 0.7126, rtol=1e-3)
            )
        )
        self.assertTrue(cond)

        self.assertTrue(np.isclose(best_fitness, -1.0316, rtol=1e-3))

    def test_schwefel_fn(self):

        def fitness_fn(x):
            """
            Schwefel Function
            https://www.sfu.ca/~ssurjano/schwef.html
            """
            dimension = x.shape[1]
            return 418.9829 * dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

        bounds = [[-500, 500]] * 4

        abc = ABC(
            population_size=50,
            fitness_fn=fitness_fn,
            init_fn=uniform_init(bounds),
            enforce_bounds=bounds,
        )
        abc.search(100)

        x1, x2, x3, x4 = abc.best_solution()

        # Assert global minimum has been reached
        cond = (
            np.isclose(x1, 420.9687, rtol=1e-3) and
            np.isclose(x2, 420.9687, rtol=1e-3) and
            np.isclose(x3, 420.9687, rtol=1e-3) and
            np.isclose(x4, 420.9687, rtol=1e-3)
        )
        self.assertTrue(cond)


if __name__ == '__main__':
    unittest.main()
