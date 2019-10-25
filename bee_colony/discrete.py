import numpy as np

from .base import AbstractABC
from .utils import get_random_other_index


class TSPSolver(AbstractABC):
    """
    Traveling Salesman Problem Solver using ABC.
    Solutions are updated by randomly swapping a pair of cities.
    """

    def update_solutions(self, solution_indices_to_update):
        new_solutions = []
        for i in solution_indices_to_update:
            solution_i = self.solutions[i]
            new_solution = randomly_swap_pair_of_cities(solution_i)
            new_solutions.append(new_solution)

        return np.stack(new_solutions)


def randomly_swap_pair_of_cities(solution):
    num_indices = len(solution)
    i = np.random.randint(0, num_indices)
    j = get_random_other_index(i, num_indices)

    new_solution = np.copy(solution)

    aux = new_solution[i]
    new_solution[i] = new_solution[j]
    new_solution[j] = aux

    return new_solution
