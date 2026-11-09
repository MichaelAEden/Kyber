# TODO: consider generalizing NTT algorithms and using examples with
#  polynomials of smaller degree and smaller coefficients.
import unittest

import numpy as np

from math_.ntt import (
    get_zeta_values_for_ntt,
    ntt,
    inverse_ntt,
    get_zeta_values_for_ntt_mul,
)
from kyber.sampling import sample_polynomial


class TestGetZetaValuesForNTT(unittest.TestCase):

    def test(self):
        # See Appendix A in FIPS 203.
        # https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
        zeta_values_actual = get_zeta_values_for_ntt()
        zeta_values_expected = np.array([1, 1729, 2580])
        np.testing.assert_array_equal(zeta_values_expected, zeta_values_actual[:3])


class TestGetZetaValuesForNTTMul(unittest.TestCase):

    def test(self):
        # See Appendix A in FIPS 203.
        # https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
        zeta_values_actual = get_zeta_values_for_ntt_mul()
        # Note the 2nd value listed in Appendix A is -17, but this is congruent
        # to 3312 (mod 3329).
        zeta_values_expected = np.array([17, 3312, 2761])
        np.testing.assert_array_equal(zeta_values_expected, zeta_values_actual[:3])


class TestNTT(unittest.TestCase):

    def test_ntt_then_inverse_ntt_is_identity(self):
        """
        NTT followed by inverse NTT should yield the same polynomial.
        """
        f_original = sample_polynomial(seed=b"\x02" * 128, nu=2, q=3329, n=256)
        f_actual = inverse_ntt(ntt(f_original))

        self.assertEqual(f_actual, f_original)


class TestNTTMultiplication(unittest.TestCase):

    def test_ntt_mul_same_as_polynomial_mul(self):
        f = sample_polynomial(seed=b"\x02" * 128, nu=2, q=3329, n=256)
        g = sample_polynomial(seed=b"\x04" * 128, nu=2, q=3329, n=256)

        h_expected = f * g
        h_actual = inverse_ntt(ntt(f) * ntt(g))

        self.assertEqual(h_actual, h_expected)
