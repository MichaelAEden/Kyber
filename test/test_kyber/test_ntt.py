import unittest

import numpy as np

from math_.ntt import compute_zeta_values, ntt, inverse_ntt
from kyber.sampling import sample_polynomial


class TestComputeZetaValues(unittest.TestCase):

    def test(self):
        # See Appendix A in FIPS 203.
        # https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
        zeta_values_actual = compute_zeta_values()
        zeta_values_expected = np.array([1, 1729, 2580])
        np.testing.assert_array_equal(zeta_values_expected, zeta_values_actual[:3])


class TestNTT(unittest.TestCase):

    def test_ntt_then_inverse_ntt_is_identity(self):
        """
        NTT followed by inverse NTT should yield the same polynomial.
        """
        # TODO: consider generalizing algorithm and using examples with
        #  polynomials of smaller degree and smaller coefficients.
        f_original = sample_polynomial(seed=b"\x02" * 128, nu=2, q=3329, n=256)
        f_temp = ntt(f_original)
        f_actual = inverse_ntt(f_temp)

        np.testing.assert_array_equal(f_actual, f_original)
