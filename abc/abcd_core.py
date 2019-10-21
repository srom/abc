import numpy as np
import tensorflow as tf


class ABCD(object):
    """
    Artificial Bee Colony algorithm for Discrete optimisation (ABCD)

    This algorithm is a flavor of the ABC algorithm [1], adapted for discrete optimisation problems.

    The main differences with the original algorithm are:
      - Solutions initialization is problem specific and left to the user
      - Updates are recombinations of existing solutions by swapping a number of dimensions.
        Half of the dimensions are swapped at each update.

    [1] Karaboga, Dervis. An idea based on honey bee swarm for numerical optimization.
        Vol. 200. Technical report-tr06, Erciyes university, engineering faculty,
        computer engineering department, 2005.
    """

    def __init__(self, population_size, fitness_fn, init_fn):
        """
        Args:
          population_size
            Number of "worker bees" to use during the search. The algorithm will keep track
            of this many solutions.

          fitness_fn
            Function to be minimized, with following signature:
            ```
            Args:
              x: solutions in flight, a tf.Tensor of shape (population_size, solution_dimension)

            Returns:
              Fitness evaluations: tf.Tensor of shape (population_size,)
            ```

          init_fn
            Function returning the set of initial solutions, with signature:
            ```
            Args:
              population_size
                Number of new solutions to initialize

            Returns:
              Initial set of solutions, a tf.Tensor of shape (population_size, solution_dimension)
            ```
        """
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.init_fn = init_fn
        self._initialized = False

    def init(self):
        self._initialized = True
        return self

    def reset(self):
        self._initialized = False
        return self.init()

    def best_solution(self):
        return None

    def best_fitness(self):
        return np.inf

    def search(self, max_generations):
        if not self._initialized:
            self.init()

        return self.best_solution(), self.best_fitness()
