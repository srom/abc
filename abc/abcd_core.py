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

    def __init__(self, population_size, fitness_fn, init_fn, scouting_threshold=None):
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

          scouting_threshold
            Number of updates without improvement after which a solution is replaced by a new one.
            Defaults to population_size * dimension.
        """
        if not np.isscalar(self.population_size) or self.population_size < 2:
            raise ValueError(f'Population size must be a number greater or equal to 2')
        elif not callable(fitness_fn):
            raise ValueError(f'Fitness function function must be callable')
        elif not callable(init_fn):
            raise ValueError(f'Init function function must be callable')
        elif scouting_threshold is not None and scouting_threshold < 2:
            raise ValueError(f'Scouting threshold must be a number greater or equal to 2')

        self.generation = 0
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.init_fn = init_fn
        self.scouting_threshold = scouting_threshold
        self._initialized = False

    def init(self):
        if self._initialized:
            raise ValueError('Already initialized - call reset method to start over')

        solutions = self.init_fn(self.population_size)

        shape = tf.shape(solutions).numpy()
        num_solutions = shape[0]
        self.dimension = shape[1]

        if num_solutions != self.population_size:
            raise ValueError(f'Expected {self.population_size} solutions but got {num_solutions}')

        self.solutions = tf.Variable(solutions)
        self.fitness_evaluations = tf.Variable(self.fitness_fn(solutions))
        self.ordered_indices = tf.Variable(tf.argsort(fitness_evaluations))
        self.no_update_counts = np.zeros(self.population_size, dtype=np.int32)

        if self.scouting_threshold is None:
            self.scouting_threshold = self.population_size * self.dimension

        self.generation = 0
        self._initialized = True
        return self

    def reset(self):
        self._initialized = False
        return self.init()

    def best_solution(self):
        return self.solutions[self.ordered_indices[0]].read_value().numpy()

    def best_fitness(self):
        return self.fitness_fn(self.solutions[self.ordered_indices[:1]])[0].numpy()

    def search(self, max_generations):
        if not self._initialized:
            self.init()

        for _ in range(max_generations):
            self.generation += 1
            self.forage_with_employed_bees()
            self.forage_with_onlooker_bees()
            self.scout_for_new_food_sources()

        return self.best_solution(), self.best_fitness()

    def forage_with_employed_bees(self):
        indices = list(range(self.population_size))
        self.update_solutions(indices)

    def forage_with_onlooker_bees(self):
        pop = tf.cast(self.population_size, tf.float64)
        weights = tf.math.log(pop + 0.5) - tf.math.log(tf.range(1, pop + 1))
        probabilities = (weights / tf.reduce_sum(weights)).numpy()

        sorted_probabilities = np.zeros((self.population_size,))
        for i, idx in enumerate(self.ordered_indices.numpy()):
            sorted_probabilities[idx] = probabilities[i]

        indices = np.random.choice(
            list(range(self.population_size)),
            p=sorted_probabilities,
            size=self.population_size,
        )
        self.update_solutions(indices)

    def scout_for_new_food_sources(self):
        """
        Solutions that haven't improved in the last 'self.limit' cycles are
        replaced by new ones, except if the solution is the best one so far.
        """
        best_idx = self.ordered_indices[0].numpy()
        new_solutions_indices = []
        for i in range(self.population_number):
            if i == best_idx:
                continue
            elif self.no_update_counts[i] > self.scouting_threshold:
                new_solutions_indices.append(i)

        if len(new_solutions_indices) > 0:
            new_solutions = self.init_fn(len(new_solutions_indices))
            new_fitness_evals = self.fitness_fn(new_solutions)

            for i in new_solutions_indices:
                self.solutions[i].assign(new_solutions[i])
                self.fitness_evaluations[i].assign(new_fitness_evals[i])
                self.no_update_counts[i] = 0

            self.ordered_indices.assign(tf.argsort(self.fitness_evaluations))

    def update_solutions(self, indices):
        new_solutions = tf.zeros((len(indices),), dtype=self.solutions.dtype)
        for i in indices:
            j = self.get_a_random_solution_index(i)
            new_solution = self.swap_dimensions(i, j)
            new_solutions[i] = new_solution

        new_fitness_evals = self.fitness_fn(new_solutions)

        for i in range(self.population_size):
            if self.fitness_evaluations[i] > new_fitness_evals[i]:
                self.solutions[i].assign(new_solutions[i])
                self.fitness_evaluations[i].assign(new_fitness_evals[i])
            else:
                self.no_update_counts[i] += 1

        self.ordered_indices.assign(tf.argsort(self.fitness_evaluations))

    def get_a_random_solution_index(self, current_solution_index):
        random_index = current_solution_index
        while random_index == current_solution_index:
            random_index = np.random.randint(0, self.population_size)

        return random_index

    def swap_dimensions(solution_index_i, solution_index_j, how_much=0.5):
        dimensions = np.random.choice(
            list(range(self.dimension)),
            int(np.ceil(how_much * self.dimension)),
        )

        solution1 = tf.identity(self.solutions[solution_index_i].read_value())
        solution2 = tf.identity(self.solutions[solution_index_j].read_value())

        solution1[dimensions] = solution2[dimensions]

        return solution1
