import unittest

import numpy as np

from kyber.sampling import sample_ntt, sample_polynomial


class TestSampleNTT(unittest.TestCase):

    def test(self):
        q = 3329
        seed = b"\x01\x02" * 17
        a = sample_ntt(seed, q=q, n=256)

        # Ensure coefficients are within valid bounds.
        np.testing.assert_array_less(a.coef, q)
        np.testing.assert_array_less(-1, a.coef)

    def test_invalid_seed_length(self):
        seed = b"\x01\x02" * 16
        with self.assertRaises(ValueError):
            sample_ntt(seed, q=3329, n=256)


class TestSamplePolynomial(unittest.TestCase):

    def test(self):
        q = 3329
        seed = b"\x01\x02" * 96
        f = sample_polynomial(seed, nu=3, q=q, n=256)

        # Ensure coefficients are within valid bounds.
        np.testing.assert_array_less(f.coef, q)
        np.testing.assert_array_less(-1, f.coef)

    def test_invalid_seed_length(self):
        seed = b"\x01\x02" * 95
        with self.assertRaises(ValueError):
            sample_polynomial(seed, nu=3, q=3329, n=256)

    def test_invalid_nu(self):
        seed = b"\x01\x02" * 128
        with self.assertRaises(ValueError):
            sample_polynomial(seed, nu=4, q=3329, n=256)
