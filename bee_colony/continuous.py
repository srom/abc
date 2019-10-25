import numpy as np

from .base import AbstractABC
from .utils import get_random_other_index


class ABC(AbstractABC):
    """
    Artificial Bee Colony (ABC) implementation as defined in [1].

    [1] Karaboga, Dervis. An idea based on honey bee swarm for numerical optimization.
        Vol. 200. Technical report-tr06, Erciyes university, engineering faculty,
        computer engineering department, 2005.
    """

    def __init__(
        self,
        population_size,
        fitness_fn,
        init_fn,
        scouting_threshold=None,
        enforce_bounds=None
    ):
        """
        Args:
          population_size
            Number of "worker bees" to use during the search. The algorithm will keep track
            of this many solutions.

          fitness_fn
            Function to be minimized, with following signature:
            ```
            Args:
              x: solutions in flight, np.ndarray of shape (population_size, solution_dimension)

            Returns:
              Fitness evaluations: np.ndarray of shape (population_size,)
            ```

          init_fn
            Function returning the set of initial solutions, with signature:
            ```
            Args:
              population_size
                Number of new solutions to initialize

            Returns:
              Initial set of solutions, np.ndarray of shape (population_size, solution_dimension)
            ```

          scouting_threshold
            Number of updates without improvement after which a solution is replaced by a new one.
            Defaults to population_size * dimension.

          enforce_bounds
            2D list, the min and max for each dimension, e.g. [[-1, 1], [None, 2], [0, 1]].
            Ensures the fitness function is never called with out of bounds values.
            Defaults to not enforcing bounds.
        """
        super().__init__(population_size, fitness_fn, init_fn, scouting_threshold)

        if enforce_bounds is not None and not isinstance(enforce_bounds, (np.ndarray, list)):
            raise ValueError('Bounds must be a list or numpy array')
        elif enforce_bounds is not None and np.ndim(enforce_bounds) != 2:
            ndim = np.ndim(enforce_bounds)
            raise ValueError(f'Bounds must be a 2D array but got an array of dim {ndim}')

        self._enforce_bounds = False
        if enforce_bounds is not None:
            self._enforce_bounds = True
            self._clip_min = np.array([m[0] for m in enforce_bounds])
            self._clip_max = np.array([m[1] for m in enforce_bounds])

    def update_solutions(self, solution_indices_to_update):
        new_solutions = []
        for idx in solution_indices_to_update:
            solution = self.solutions[idx]

            dim_i = np.random.randint(0, self.dimension)
            sol_j = get_random_other_index(idx, self.population_size)

            new_solution = np.copy(solution)
            other_solution = self.solutions[sol_j]

            eps = np.random.uniform(-1, 1)
            new_solution[dim_i] += eps * (solution[dim_i] - other_solution[dim_i])

            new_solutions.append(new_solution)

        if self._enforce_bounds:
            return np.clip(new_solutions, self._clip_min, self._clip_max)
        else:
            return np.stack(new_solutions)
