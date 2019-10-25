import unittest

import numpy as np

from .utils import get_random_other_index, assign_probabilities


class TestUtilsFunctions(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)

        np.random.seed(123)

    def test_get_random_solution_index(self):
        self.assertNotEqual(get_random_other_index(1, 5), 1)
        self.assertNotEqual(get_random_other_index(2, 5), 2)
        self.assertNotEqual(get_random_other_index(0, 8), 0)

    def test_assign_probabilities(self):
        probabilities = [0.6, 0.3, 0.1]
        ordered_indices = [2, 0, 1]

        sorted_probabilities = assign_probabilities(probabilities, ordered_indices)

        self.assertEqual(
            [0.3, 0.1, 0.6],
            sorted_probabilities.tolist(),
        )

        with self.assertRaises(ValueError):
            assign_probabilities(probabilities, ordered_indices[1:])


if __name__ == '__main__':
    unittest.main()
