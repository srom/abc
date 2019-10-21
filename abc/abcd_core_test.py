import unittest

import numpy as np
import tensorflow as tf


class TestABCD(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)

        np.random.seed(123)
        tf.random.set_seed(123)

    def test(self):
        pass


if __name__ == '__main__':
    unittest.main()
