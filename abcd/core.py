import numpy as np
import tensorflow as tf


class ABCD(object):
    """
    Artificial Bee Colony algorithm for Discrete optimisation (ABCD)

    This algorithm is a flavor of the ABC algorithm [1], adapted for discrete optimisation problems.

    The main differences with the original algorithm are:
      - Solutions initialization is problem specific and left to the user
      - Updates are recombinations of existing solutions by swapping a number of dimensions.
        Half of the dimensions are swapped at each step.

    [1] Karaboga, Dervis. An idea based on honey bee swarm for numerical optimization.
        Vol. 200. Technical report-tr06, Erciyes university, engineering faculty,
        computer engineering department, 2005.
    """

    def __init__(self, population_size, fitness_fn, init_fn):
        """
        Args:
          population_size
            Number of solutions to keep track of.

          fitness_fn
            Function to be minimized, with following signature:
            ```
            Args:
              x: solutions in flight, a tf.Tensor of shape (population_size, num_dimensions)

            Returns:
              Fitness evaluations: tf.Tensor of shape (population_size,)
            ```

          init_fn
            Function returning the set of initial solutions, with signature:
            ```
            Args:
              population_size

            Returns:
              An initial set of solutions, a tf.Tensor of shape (population_size, num_dimensions)
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
