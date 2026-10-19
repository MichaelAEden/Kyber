import unittest

import numpy as np

from math_.modulo import modular_rounding


class TestModularRounding(unittest.TestCase):
    # See Definition 4.12 in the following document:
    # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf

    def test_round_to_zero(self):
        q = 1223
        x = np.array([-305, 0, 305])

        y = modular_rounding(x, q=q)

        np.testing.assert_array_equal(y, np.zeros(3))

    def test_round_to_zero_with_mod(self):
        q = 1223
        x = np.array([-305, 0, 305]) + q

        y = modular_rounding(x, q=q)

        np.testing.assert_array_equal(y, np.zeros(3))

    def test_round_to_one(self):
        q = 1223
        x = np.array([-306, -611, 611, 306])

        y = modular_rounding(x, q=q)

        np.testing.assert_array_equal(y, np.ones(4))

    def test_round_to_one_with_mod(self):
        q = 1223
        x = np.array([-306, -611, 611, 306]) + q

        y = modular_rounding(x, q=q)

        np.testing.assert_array_equal(y, np.ones(4))
