import unittest

import numpy as np

from encryption.lindner_peikert import LindnerPeikert


class TestLindnerPeikert(unittest.TestCase):

    def test_random_values(self):
        x = np.array([0, 1, 0, 1])

        lp = LindnerPeikert(
            n=4,
            q=997,
            B=5,
        )
        u, v = lp.encrypt(x=x)
        x_decrypted = lp.decrypt(u, v)

        self.assertEqual((4,), x_decrypted.shape)
        np.testing.assert_array_equal(x_decrypted, x)

    def test_fixed_values(self):
        # See Example 4.14 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        x = np.array([0])

        lp = LindnerPeikert(
            n=4,
            q=997,
            B=5,
            A=np.array(
                [
                    [301, 352, 400, 269],
                    [585, 916, 438, 560],
                    [510, 108, 14, 157],
                    [265, 620, 886, 348],
                ]
            ),
            s=np.array([2, 3, -5, 1]).reshape((4, 1)),
            e=np.array([4, 5, 0, 1]).reshape((4, 1)),
        )
        u, v = lp.encrypt(
            x=x,
            r=np.array([-3, -3, 4, -5]).reshape((4, 1)),
            e1=np.array([-1, 2, -3, -5]).reshape((4, 1)),
            e2=np.array([-3]).reshape((1, 1)),
        )
        x_decrypted = lp.decrypt(u, v)

        self.assertEqual((1,), x_decrypted.shape)
        np.testing.assert_array_equal(x_decrypted, np.array([0]))
        np.testing.assert_array_equal(
            lp.b, np.array([928, 299, 434, 303]).reshape((4, 1))
        )
        np.testing.assert_array_equal(u, np.array([50, 509, 88, 384]).reshape((4, 1)))
        np.testing.assert_array_equal(v, np.array([525]).reshape((1, 1)))
