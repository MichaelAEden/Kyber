import unittest

import numpy as np

from kyber.compression import compress, decompress, round_fraction


class TestRoundFraction(unittest.TestCase):

    def test(self):
        numerator = np.array([0, 5, 6, 11])
        denominator = 11
        y_actual = round_fraction(numerator, denominator)
        y_expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(y_actual, y_expected)


class TestCompress(unittest.TestCase):

    def test(self):
        # See Example 6.9 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        np.testing.assert_array_equal(
            np.array([746]), compress(x=np.array([2426]), q=3329, d=10)
        )


class TestDecompress(unittest.TestCase):

    def test(self):
        # See Example 6.9 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        np.testing.assert_array_equal(
            np.array([2425]), decompress(y=np.array([746]), q=3329, d=10)
        )
