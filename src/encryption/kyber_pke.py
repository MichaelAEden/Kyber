import numpy as np

from encryption.kyber_key_gen import random_polynomial_matrix
from math_.matrix import Matrix
from math_.modulo import modular_rounding
from math_.polynomial_ring import PolynomialRing

KyberCiphertext = tuple[Matrix[PolynomialRing], Matrix[PolynomialRing]]


class KyberPKE:
    """
    Implements Kyber public-key encryption (PKE) scheme, without ciphertext compression.

    See Section 6.1 in the following document:
    https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
    """

    # TODO: is cyphertext compression lossy or does it only increase decryption error rate?

    def __init__(
        self,
        n: int,
        q: int,
        k: int,
        nu1: int,
        nu2: int,
        A: Matrix[PolynomialRing] = None,
        s: Matrix[PolynomialRing] = None,
        e: Matrix[PolynomialRing] = None,
        seed: int = None,
    ):
        """
        :param n: Degree of reduction polynomial.
        :param q: Prime modulus.
        :param k: Dimensions of lattice matrix.
        :param nu1: Coefficient bound for s, e, r.
        :param nu2: Coefficient bound for e1, e2.
        :param seed: Seed for RNG.
        """
        rng = np.random.default_rng(seed)

        # Parameters
        self.n = n
        self.q = q
        self.k = k
        self.nu1 = nu1
        self.nu2 = nu2

        self.A = (
            A
            if A is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.q,
                row_count=self.k,
                col_count=self.k,
                rng=rng,
            )
        )
        self.s = (
            s
            if s is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=rng,
            )
        )
        e = (
            e
            if e is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=rng,
            )
        )
        self.t = self.A * self.s + e

    @property
    def public_key(self) -> tuple[Matrix[PolynomialRing], Matrix[PolynomialRing]]:
        return self.A, self.t

    @property
    def private_key(self) -> Matrix[PolynomialRing]:
        return self.s

    def encrypt(
        self,
        m: np.ndarray,
        r: Matrix[PolynomialRing] = None,
        e1: Matrix[PolynomialRing] = None,
        e2: Matrix[PolynomialRing] = None,
        seed: int = None,
    ) -> KyberCiphertext:
        rng = np.random.default_rng(seed)

        r = (
            r
            if r is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=rng,
            )
        )
        e1 = (
            e1
            if e1 is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu2,
                row_count=self.k,
                col_count=1,
                rng=rng,
            )
        )
        e2 = (
            e2
            if e2 is not None
            else random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu2,
                row_count=1,
                col_count=1,
                rng=rng,
            )
        )

        u = self.A.transpose * r + e1
        # Use ceil instead of round such that ties are broken upwards.
        v = (
            self.t.transpose * r
            + e2
            + Matrix.scalar(
                PolynomialRing(q=self.q, n=self.n, coef=np.ceil(self.q / 2) * m)
            )
        )

        return u, v

    def decrypt(
        self,
        c: KyberCiphertext,
    ) -> np.ndarray:
        u, v = c
        m = modular_rounding((v - self.s.transpose * u)[0, 0].coef, q=self.q)
        return m.astype(bool).flatten()
