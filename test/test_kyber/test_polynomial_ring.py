import unittest

import numpy as np

from kyber.polynomial_ring import PolynomialRing


class TestPolynomialRing(unittest.TestCase):
    # See Example 5.1 in the following document:
    # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf

    def test_multiplication(self):
        q = 43
        n = 4
        f = PolynomialRing(q=q, n=n, x=np.array([33, 41, 0, 12]))
        g = PolynomialRing(q=q, n=n, x=np.array([2, 17, 39, 14]))

        h = f * g

        self.assertEqual(q, h.q)
        self.assertEqual(n, h.n)
        np.testing.assert_array_equal(np.array([19, 3, 10, 21]), h.x)

    def test_multiplication_incompatible(self):
        f = PolynomialRing(q=43, n=4, x=np.array([33, 41, 0, 12]))
        g = PolynomialRing(q=47, n=4, x=np.array([2, 17, 39, 14]))

        with self.assertRaises(ValueError):
            f * g

    def test_addition(self):
        q = 43
        n = 4
        f = PolynomialRing(q=q, n=n, x=np.array([33, 41, 0, 12]))
        g = PolynomialRing(q=q, n=n, x=np.array([2, 17, 39, 14]))

        h = f + g

        self.assertEqual(q, h.q)
        self.assertEqual(n, h.n)
        np.testing.assert_array_equal(np.array([35, 15, 39, 26]), h.x)

    def test_subtraction(self):
        q = 43
        n = 4
        f = PolynomialRing(q=q, n=n, x=np.array([33, 41, 0, 12]))
        g = PolynomialRing(q=q, n=n, x=np.array([2, 17, 39, 14]))

        h = f - g

        self.assertEqual(q, h.q)
        self.assertEqual(n, h.n)
        np.testing.assert_array_equal(np.array([31, 24, 4, 41]), h.x)


if __name__ == '__main__':
    unittest.main()
