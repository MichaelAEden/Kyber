import unittest

import numpy as np

from kyber.pke import KyberPKE
from math_.matrix import Matrix
from math_.polynomial_ring import Polynomial


class TestKyberPKE(unittest.TestCase):

    def test_random_values(self):
        q = 3329
        n = 256

        m = b"\x00\x12\x34\x56" * 8

        kyber = KyberPKE.from_seed(
            n=n,
            q=q,
            k=3,
            nu1=3,
            nu2=3,
            du=10,
            dv=4,
            seed=b"\x00" * 32,
        )
        c = kyber.encrypt(m, seed=b"\x00" * 32)
        m_decrypted = kyber.decrypt(c)

        self.assertEqual(n // 8, len(m_decrypted))
        self.assertEqual(m, m_decrypted)

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
            rho=b"\x00" * 32,
            A_ntt=Matrix(
                (
                    (
                        Polynomial(q=q, n=n, coef=np.array([329, 26, 34])),
                        Polynomial(q=q, n=n, coef=np.array([295, 745, 292])),
                        Polynomial(q=q, n=n, coef=np.array([732, 248, 641])),
                    ),
                    (
                        Polynomial(q=q, n=n, coef=np.array([833, 849, 647])),
                        Polynomial(q=q, n=n, coef=np.array([396, 363, 924])),
                        Polynomial(q=q, n=n, coef=np.array([312, 803, 31])),
                    ),
                    (
                        Polynomial(q=q, n=n, coef=np.array([176, 501, 802])),
                        Polynomial(q=q, n=n, coef=np.array([738, 941, 521])),
                        Polynomial(q=q, n=n, coef=np.array([19, 582, 224])),
                    ),
                )
            ),
            s_ntt=Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([1, 0, 3])),
                    Polynomial(q=q, n=n, coef=np.array([1, 1, 0])),
                    Polynomial(q=q, n=n, coef=np.array([-2, 0, 2])),
                ),
                row_vector=False,
            ),
            e_ntt=Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([-2, 3, -1])),
                    Polynomial(q=q, n=n, coef=np.array([1, -1, 2])),
                    Polynomial(q=q, n=n, coef=np.array([3, 1, -1])),
                ),
                row_vector=False,
            ),
        )
        c = kyber._encrypt(
            m=m,
            y=Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([-2, -2, 1])),
                    Polynomial(q=q, n=n, coef=np.array([0, 0, 3])),
                    Polynomial(q=q, n=n, coef=np.array([-1, -1, 3])),
                ),
                row_vector=False,
            ),
            e1=Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([3, -2, 2])),
                    Polynomial(q=q, n=n, coef=np.array([2, -3, -1])),
                    Polynomial(q=q, n=n, coef=np.array([-3, 1, -2])),
                ),
                row_vector=False,
            ),
            e2=Matrix.scalar(Polynomial(q=q, n=n, coef=np.array([3, -3, 2]))),
        )
        m_decrypted = kyber.decrypt(c)

        self.assertEqual((3,), m_decrypted.shape)
        self.assertEqual(
            kyber.t_ntt,
            Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([274, 180, 257])),
                    Polynomial(q=q, n=n, coef=np.array([484, 971, 42])),
                    Polynomial(q=q, n=n, coef=np.array([664, 145, 399])),
                ),
                row_vector=False,
            ),
        )
        u, v = c
        self.assertEqual(
            u,
            Matrix.vector(
                (
                    Polynomial(q=q, n=n, coef=np.array([918, 176, 944])),
                    Polynomial(q=q, n=n, coef=np.array([77, 530, 160])),
                    Polynomial(q=q, n=n, coef=np.array([572, 989, 130])),
                ),
                row_vector=False,
            ),
        )
        self.assertEqual(
            v,
            Matrix.scalar(
                Polynomial(q=q, n=n, coef=np.array([636, 169, 320])),
            ),
        )
        np.testing.assert_array_equal(m_decrypted, m)
