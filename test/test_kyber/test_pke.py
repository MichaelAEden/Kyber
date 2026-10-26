import unittest

import numpy as np

from kyber.pke import KyberPKE
from math_.matrix import Matrix
from math_.polynomial_ring import PolynomialRing


class TestKyberPKE(unittest.TestCase):

    def test_random_values(self):
        m = np.array([1, 1, 0])

        q = 991
        n = 3
        kyber = KyberPKE(
            n=n,
            q=q,
            k=3,
            nu1=3,
            nu2=3,
            seed=2026,
        )
        c = kyber.encrypt(m, seed=2027)
        m_decrypted = kyber.decrypt(c)

        self.assertEqual((3,), m_decrypted.shape)
        np.testing.assert_array_equal(m_decrypted, m)

    def test_fixed_values(self):
        # See Example 6.2 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        m = np.array([1, 1, 0])

        q = 991
        n = 3
        kyber = KyberPKE(
            n=n,
            q=q,
            k=3,
            nu1=3,
            nu2=3,
            A=Matrix(
                (
                    (
                        PolynomialRing(q=q, n=n, coef=np.array([329, 26, 34])),
                        PolynomialRing(q=q, n=n, coef=np.array([295, 745, 292])),
                        PolynomialRing(q=q, n=n, coef=np.array([732, 248, 641])),
                    ),
                    (
                        PolynomialRing(q=q, n=n, coef=np.array([833, 849, 647])),
                        PolynomialRing(q=q, n=n, coef=np.array([396, 363, 924])),
                        PolynomialRing(q=q, n=n, coef=np.array([312, 803, 31])),
                    ),
                    (
                        PolynomialRing(q=q, n=n, coef=np.array([176, 501, 802])),
                        PolynomialRing(q=q, n=n, coef=np.array([738, 941, 521])),
                        PolynomialRing(q=q, n=n, coef=np.array([19, 582, 224])),
                    ),
                )
            ),
            s=Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([1, 0, 3])),
                    PolynomialRing(q=q, n=n, coef=np.array([1, 1, 0])),
                    PolynomialRing(q=q, n=n, coef=np.array([-2, 0, 2])),
                ),
                row_vector=False,
            ),
            e=Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([-2, 3, -1])),
                    PolynomialRing(q=q, n=n, coef=np.array([1, -1, 2])),
                    PolynomialRing(q=q, n=n, coef=np.array([3, 1, -1])),
                ),
                row_vector=False,
            ),
        )
        c = kyber.encrypt(
            m=m,
            r=Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([-2, -2, 1])),
                    PolynomialRing(q=q, n=n, coef=np.array([0, 0, 3])),
                    PolynomialRing(q=q, n=n, coef=np.array([-1, -1, 3])),
                ),
                row_vector=False,
            ),
            e1=Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([3, -2, 2])),
                    PolynomialRing(q=q, n=n, coef=np.array([2, -3, -1])),
                    PolynomialRing(q=q, n=n, coef=np.array([-3, 1, -2])),
                ),
                row_vector=False,
            ),
            e2=Matrix.scalar(PolynomialRing(q=q, n=n, coef=np.array([3, -3, 2]))),
        )
        m_decrypted = kyber.decrypt(c)

        self.assertEqual((3,), m_decrypted.shape)
        self.assertEqual(
            kyber.t,
            Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([274, 180, 257])),
                    PolynomialRing(q=q, n=n, coef=np.array([484, 971, 42])),
                    PolynomialRing(q=q, n=n, coef=np.array([664, 145, 399])),
                ),
                row_vector=False,
            ),
        )
        u, v = c
        self.assertEqual(
            u,
            Matrix.vector(
                (
                    PolynomialRing(q=q, n=n, coef=np.array([918, 176, 944])),
                    PolynomialRing(q=q, n=n, coef=np.array([77, 530, 160])),
                    PolynomialRing(q=q, n=n, coef=np.array([572, 989, 130])),
                ),
                row_vector=False,
            ),
        )
        self.assertEqual(
            v,
            Matrix.scalar(
                PolynomialRing(q=q, n=n, coef=np.array([636, 169, 320])),
            ),
        )
        np.testing.assert_array_equal(m_decrypted, m)
