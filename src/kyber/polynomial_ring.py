import numpy as np
from scipy.linalg import circulant


def anti_circulant(x) -> np.ndarray:
    """
    Generate the anti-circulant matrix for the given vector.

    See Section 5.1 in the following document:
    https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
    """
    n = len(x)
    mask = np.ones((n, n)) - 2 * np.triu(np.ones((n, n)), k=1)
    return circulant(x) * mask


class PolynomialRing:

    def __init__(self, q: int, n: int, coef: np.ndarray):
        """
        :param q: Maximum value for polynomial coefficient.
        :param n: Degree of reduction polynomial.
        :param coef: Polynomial coefficients in order of ascending degree.
        """
        if len(coef) != n:
            raise ValueError("Polynomial coefficients must have length n.")

        self.q = q
        self.n = n
        self.coef = coef

    def _check_compatibility(self, other: "PolynomialRing"):
        if self.q != other.q:
            raise ValueError("Polynomials must have same q.")
        if self.n != other.n:
            raise ValueError("Polynomials must have same n.")

    def __mul__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        a = anti_circulant(self.coef)
        b = other.coef.reshape((self.n, 1))

        c = np.matmul(a, b).reshape((self.n,))
        c = np.mod(c, self.q)

        return PolynomialRing(q=self.q, n=self.n, coef=c)

    def __add__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        x = self.coef + other.coef
        x = np.mod(x, self.q)

        return PolynomialRing(q=self.q, n=self.n, coef=x)

    def __sub__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        x = self.coef - other.coef
        x = np.mod(x, self.q)

        return PolynomialRing(q=self.q, n=self.n, coef=x)
