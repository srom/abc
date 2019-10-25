import unittest

import numpy as np
import tsplib95

from .discrete import (
    TSPSolver,
    get_random_other_index,
)


class TestTSPSolver(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)

        np.random.seed(123)

    def test_djibouti(self):
        """
        Djibouti - 38 Cities
        http://www.math.uwaterloo.ca/tsp/world/countries.html
        """
        problem = tsplib95.load_problem('fixtures/djibouti.tsp')

        self.assertEqual(38, problem.dimension)

        nodes = list(problem.get_nodes())

        distance_matrix = []
        for a in nodes:
            row = []
            for b in nodes:
                row.append(problem.wfunc(a, b))

            distance_matrix.append(row)

        distance_matrix = np.array(distance_matrix)

        def init_fn(population_size):
            inits = []
            for _ in range(population_size):
                inits.append(np.random.choice(nodes, size=len(nodes), replace=False).tolist())
            return np.stack(inits)

        def fitness_fn(x):
            population_size = len(x)
            fitness_values = []
            for i in range(population_size):
                xi = x[i]
                fitness = np.sum([distance_matrix[a - 1, b - 1] for a, b in zip(xi, xi[1:])])
                fitness += distance_matrix[xi[-1] - 1, xi[0] - 1]
                fitness_values.append(fitness)
            return np.array(fitness_values)

        solver = TSPSolver(
            population_size=50,
            fitness_fn=fitness_fn,
            init_fn=init_fn,
        )

        solver.init()

        initial_fitness = solver.best_fitness()

        best_solution, best_fitness = solver.search(100)

        # Asserts that we made progress after 100 iterations
        self.assertLess(best_fitness, initial_fitness)

        # Ensures that every city appears exactly once in the solution.
        self.assertEqual(len(np.unique(best_solution)), len(nodes))


if __name__ == '__main__':
    unittest.main()
