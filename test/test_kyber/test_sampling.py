import unittest

import numpy as np

from kyber.sampling import sample_ntt


class TestSampleNTT(unittest.TestCase):

    def test(self):
        q = 3329
        seed = b"\x01\x02" * 17
        a = sample_ntt(seed, q=q)

        # Ensure coefficients are within valid bounds.
        np.testing.assert_array_less(a, q)
        np.testing.assert_array_less(-1, a)
