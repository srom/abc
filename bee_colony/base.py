import logging

import numpy as np

from .utils import assign_probabilities


logger = logging.getLogger(__name__)


class AbstractABC(object):
    """
    ABC: Artificial Bee Colony [1] base class.

    Method `update_solutions` ought to be implemented by sub classes.

    [1] Karaboga, Dervis. An idea based on honey bee swarm for numerical optimization.
        Vol. 200. Technical report-tr06, Erciyes university, engineering faculty,
        computer engineering department, 2005.
    """

    def __init__(
        self,
        population_size,
        fitness_fn,
        init_fn,
        callback_fn=None,
        scouting_threshold=None,
        termination_threshold=1e-12,
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

          callback_fn
            User defined function called first after initialization, then at the end of each
            generation. Intended for logging purpose.
            Function ought to have the following signature:
            ```
            Args:
              abc: the parent ABC instance (i.e. self)
              logger: a python Logger instance
            ```

          scouting_threshold
            Number of updates without improvement after which a solution is replaced by a new one.
            Defaults to population_size * dimension.

          termination_threshold
            Stop if |best_fitness - worse_fitness| < termination_threshold
            Defaults to 1e-12
        """
        if not np.isscalar(population_size) or population_size < 2:
            raise ValueError(f'Population size must be a number greater or equal to 2')
        elif not callable(fitness_fn):
            raise ValueError(f'Fitness function function must be callable')
        elif not callable(init_fn):
            raise ValueError(f'Init function function must be callable')
        elif scouting_threshold is not None and scouting_threshold < 2:
            raise ValueError(f'Scouting threshold must be a number greater or equal to 2')
        elif callback_fn is not None and not callable(callback_fn):
            raise ValueError(f'Callback function must be callable')

        self.generation = 0
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.init_fn = init_fn
        self.scouting_threshold = scouting_threshold
        self.termination_threshold = termination_threshold
        self.callback_fn = callback_fn
        self._initialized = False

    def init(self):
        if self._initialized:
            raise ValueError('Already initialized - call reset method to start over')

        self.solutions = self.init_fn(self.population_size)

        shape = self.solutions.shape
        num_solutions = shape[0]
        self.dimension = shape[1]

        if num_solutions != self.population_size:
            raise ValueError(f'Expected {self.population_size} solutions but got {num_solutions}')

        self.fitness_evaluations = self.fitness_fn(self.solutions)
        self.ordered_indices = np.argsort(self.fitness_evaluations)
        self.no_update_counts = np.zeros(self.population_size, dtype=np.int32)

        half_pop = int(np.ceil(self.population_size / 2))
        weights = np.log(half_pop + 0.5) - np.log(list(range(1, half_pop + 1)))
        weights = np.concatenate([weights, np.zeros((self.population_size - half_pop,))])
        self.normalized_weights = weights / np.sum(weights)

        self.population_indices = list(range(self.population_size))
        self.dimension_indices = list(range(self.dimension))

        if self.scouting_threshold is None:
            self.scouting_threshold = self.population_size * self.dimension

        self.generation = 0
        self._initialized = True
        return self

    def reset(self):
        self._initialized = False
        return self.init()

    def best_solution(self):
        return self.solutions[self.ordered_indices[0]]

    def best_fitness(self):
        return self.fitness_evaluations[self.ordered_indices[0]]

    def search(self, max_generations):
        if not self._initialized:
            self.init()

        # Call user defined function at generation 0
        if self.callback_fn is not None:
            self.callback_fn(self, logger)

        for _ in range(max_generations):
            self.generation += 1
            self.forage_with_employed_bees()
            self.forage_with_onlooker_bees()
            self.scout_for_new_food_sources()

            self.termination_criterion_met = self.should_terminate()

            # Call user defined function last
            if self.callback_fn is not None:
                self.callback_fn(self, logger)

            if self.termination_criterion_met:
                break

        return self.best_solution(), self.best_fitness()

    def forage_with_employed_bees(self):
        self.search_for_improved_solutions(self.population_indices)

    def forage_with_onlooker_bees(self):
        onlooker_probabilities = assign_probabilities(
            self.normalized_weights,
            self.ordered_indices,
        )
        indices = np.random.choice(
            self.population_indices,
            p=onlooker_probabilities,
            size=self.population_size,
        )
        self.search_for_improved_solutions(indices)

    def scout_for_new_food_sources(self):
        """
        Solutions that haven't improved in the last `scouting_threshold` updates are
        replaced by new ones, except if the solution is the best one so far.
        """
        best_idx = self.ordered_indices[0]
        new_solutions_indices = []
        for i in self.population_indices:
            if i == best_idx:
                continue
            elif self.no_update_counts[i] > self.scouting_threshold:
                new_solutions_indices.append(i)

        if len(new_solutions_indices) > 0:
            new_solutions = self.init_fn(len(new_solutions_indices))
            new_fitness_evals = self.fitness_fn(new_solutions)

            for i, idx in enumerate(new_solutions_indices):
                self.solutions[idx] = new_solutions[i]
                self.fitness_evaluations[idx] = new_fitness_evals[i]
                self.no_update_counts[idx] = 0

            self.ordered_indices = np.argsort(self.fitness_evaluations)

    def search_for_improved_solutions(self, solution_indices_to_update):
        new_solutions = self.update_solutions(solution_indices_to_update)
        new_fitness_evals = self.fitness_fn(new_solutions)

        for c, i in enumerate(solution_indices_to_update):
            if self.fitness_evaluations[i] > new_fitness_evals[c]:
                self.solutions[i] = new_solutions[c]
                self.fitness_evaluations[i] = new_fitness_evals[c]
                self.no_update_counts[i] = 0
            else:
                self.no_update_counts[i] += 1

        self.ordered_indices = np.argsort(self.fitness_evaluations)

    def update_solutions(self, solution_indices_to_update):
        msg = 'Method update_solutions is problem specific and ought to be sub-classed'
        raise NotImplementedError(msg)

    def should_terminate(self):
        best_fitness = self.fitness_evaluations[self.ordered_indices[0]]
        worse_fitness = self.fitness_evaluations[self.ordered_indices[-1]]
        return np.abs(best_fitness - worse_fitness) < self.termination_threshold
