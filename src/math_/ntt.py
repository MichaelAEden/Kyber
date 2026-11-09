"""
See Section 4.3 of FIPS 203 regarding the Number-Theoretic Transform (NTT).
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

from dataclasses import dataclass

import numpy as np

from kyber.byte_encoding import bit_reverse
from math_.polynomial_ring import Polynomial


def get_zeta_values_for_ntt():
    # TODO: replace with Appendix A or cache.
    q = 3329
    n = 128
    zeta = 17
    powers = bit_reverse(np.arange(n), d=7)
    return np.array([pow(zeta, int(p), q) for p in powers], dtype=np.uint16)


def get_zeta_values_for_ntt_mul():
    # TODO: replace with Appendix A or cache.
    q = 3329
    n = 128
    zeta = 17
    powers = 2 * bit_reverse(np.arange(n), d=7) + 1
    return np.array([pow(zeta, int(p), q) for p in powers], dtype=np.uint16)


def ntt(p: Polynomial) -> "PolynomialNTT":
    """Convert polynomial to its NTT representation."""
    q = 3329
    n = 256

    if p.q != q or p.n != n:
        raise ValueError(f"NTT only implemented for q={q}, n={n}.")

    f = np.astype(p.coef, np.int16)
    zeta_values = get_zeta_values_for_ntt()

    i = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = zeta_values[i]
            i += 1
            j = start
            while j < start + length:
                t = (zeta * f[j + length]) % q
                f[j + length] = (f[j] - t) % q
                f[j] = (f[j] + t) % q
                j += 1
            start = start + 2 * length
        length = length // 2

    return PolynomialNTT(q=q, n=n, coef=f.astype(np.uint16))


def inverse_ntt(p: "PolynomialNTT") -> Polynomial:
    """Convert NTT representation back to polynomial."""
    q = 3329
    n = 256

    if p.q != q or p.n != n:
        raise ValueError(f"NTT only implemented for q={q}, n={n}.")

    f = np.astype(p.coef, np.int16)
    zeta_values = get_zeta_values_for_ntt()

    i = 127
    length = 2
    while length <= 128:
        start = 0
        while start < 256:
            zeta = zeta_values[i]
            i -= 1
            j = start
            while j < start + length:
                t = f[j]
                f[j] = (t + f[j + length]) % q
                f[j + length] = (zeta * (f[j + length] - t)) % q
                j += 1
            start = start + 2 * length
        length = length * 2
    f = (f.astype(np.int64) * 3303) % q

    return Polynomial(q=q, n=n, coef=f.astype(np.uint16))


def _base_case_multiply(
    a0: int, a1: int, b0: int, b1: int, gamma: int, q: int
) -> tuple[int, int]:
    c0 = (a0 * b0 + a1 * b1 * gamma) % q
    c1 = (a0 * b1 + a1 * b0) % q
    return c0, c1


@dataclass(frozen=True)
class PolynomialNTT:
    # Maximum value for polynomial coefficient.
    q: int
    # Degree of reduction polynomial.
    n: int
    # Polynomial coefficients in order of ascending degree.
    coef: np.ndarray

    def __post_init__(self):
        if len(self.coef) != self.n:
            raise ValueError(f"Polynomial coefficients must have length n={self.n}.")
        if self.coef.dtype != np.uint16:
            raise ValueError(f"Polynomial coefficients must be uint16.")
        if np.any(self.coef >= self.q):
            raise ValueError(f"Polynomial coefficients must be less than q={self.q}.")

    def _check_compatibility(self, other):
        if not isinstance(other, PolynomialNTT):
            raise TypeError(
                f"Cannot perform operation between PolynomialNTT and {type(other)}."
            )
        if self.q != other.q:
            raise ValueError("Polynomials must have same q.")
        if self.n != other.n:
            raise ValueError("Polynomials must have same n.")

    def __eq__(self, other) -> bool:
        if not isinstance(other, PolynomialNTT):
            return False

        return (
            self.q == other.q
            and self.n == other.n
            and np.array_equal(self.coef, other.coef)
        )

    def __mul__(self, other) -> "PolynomialNTT":
        self._check_compatibility(other)

        n = 256
        if self.n != n:
            raise ValueError(f"Multiplication only supported for n = {n}")

        zeta_values = get_zeta_values_for_ntt_mul()
        f = self.coef
        g = other.coef
        h = np.zeros_like(self.coef)
        for i in range(128):
            h[2 * i], h[2 * i + 1] = _base_case_multiply(
                int(f[2 * i]),
                int(f[2 * i + 1]),
                int(g[2 * i]),
                int(g[2 * i + 1]),
                gamma=int(zeta_values[i]),
                q=self.q,
            )

        return PolynomialNTT(
            q=self.q,
            n=self.n,
            coef=h,
        )

    def __add__(self, other) -> "PolynomialNTT":
        self._check_compatibility(other)

        x = self.coef + other.coef
        x = np.mod(x, self.q)

        return PolynomialNTT(q=self.q, n=self.n, coef=x)

    def __sub__(self, other) -> "PolynomialNTT":
        self._check_compatibility(other)

        x = self.coef - other.coef.astype(np.int16)
        x = np.mod(x, self.q).astype(np.uint16)

        return PolynomialNTT(q=self.q, n=self.n, coef=x)
