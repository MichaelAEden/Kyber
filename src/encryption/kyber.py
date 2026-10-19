from typing import Callable

import numpy as np

from math_.modulo import modular_rounding
from math_.matrix import Matrix
from math_.polynomial_ring import PolynomialRing


def _random_polynomial(
    q: int, n: int, max_coef: int, rng: np.random.Generator
) -> PolynomialRing:
    coef = rng.integers(low=0, high=max_coef, size=n)
    return PolynomialRing(q=q, n=n, coef=coef)


def _random_matrix(row_count: int, col_count: int, fn: Callable) -> Matrix:
    return Matrix(
        tuple(tuple(fn() for _ in range(col_count)) for _ in range(row_count))
    )


def _random_polynomial_matrix(
    q: int,
    n: int,
    max_coef: int,
    row_count: int,
    col_count: int,
    rng: np.random.Generator,
) -> Matrix[PolynomialRing]:
    return _random_matrix(
        row_count=row_count,
        col_count=col_count,
        fn=lambda: _random_polynomial(q, n, max_coef, rng),
    )


class Kyber:
    """
    Implements Kyber public-key encryption scheme, without ciphertext compression.

    See Section 6.1 in the following document:
    https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
    """

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
        self.rng = np.random.default_rng(seed)

        # Parameters
        self.n = n
        self.q = q
        self.k = k
        self.nu1 = nu1
        self.nu2 = nu2

        # Public key: (A, s)
        self.A = (
            A
            if A is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.q,
                row_count=self.k,
                col_count=self.k,
                rng=self.rng,
            )
        )
        self.s = (
            s
            if s is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=self.rng,
            )
        )

        # Private key: t
        e = (
            e
            if e is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=self.rng,
            )
        )
        t = self.A * self.s + e
        self.t = t

    def encrypt(
        self,
        x: np.ndarray,
        r: Matrix[PolynomialRing] = None,
        e1: Matrix[PolynomialRing] = None,
        e2: Matrix[PolynomialRing] = None,
    ) -> tuple[Matrix[PolynomialRing], Matrix[PolynomialRing]]:
        r = (
            r
            if r is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu1,
                row_count=self.k,
                col_count=1,
                rng=self.rng,
            )
        )
        e1 = (
            e1
            if e1 is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu2,
                row_count=self.k,
                col_count=1,
                rng=self.rng,
            )
        )
        e2 = (
            e2
            if e2 is not None
            else _random_polynomial_matrix(
                q=self.q,
                n=self.n,
                max_coef=self.nu2,
                row_count=1,
                col_count=1,
                rng=self.rng,
            )
        )

        u = self.A.transpose * r + e1
        # Use ceil instead of round such that ties are broken upwards.
        v = (
            self.t.transpose * r
            + e2
            + Matrix.scalar(
                PolynomialRing(q=self.q, n=self.n, coef=np.ceil(self.q / 2) * x)
            )
        )

        return u, v

    def decrypt(
        self, u: Matrix[PolynomialRing], v: Matrix[PolynomialRing]
    ) -> np.ndarray:
        x = modular_rounding((v - self.s.transpose * u)[0, 0].coef, q=self.q)
        return x.flatten()
