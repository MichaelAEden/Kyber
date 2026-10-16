import numpy as np


class PolynomialRing:

    def __init__(self, q: int, n: int, x: np.ndarray):
        """
        :param q: Maximum value for polynomial coefficient.
        :param n: Degree of reduction polynomial.
        :param x: Polynomial coefficients in order of ascending degree.
        """
        if len(x) != n:
            raise ValueError("Polynomial coefficients must have length n.")

        self.q = q
        self.n = n
        self.x = x

    def _reduction_polynomial_coefficients(self) -> np.ndarray:
        coeffs = np.zeros(self.n + 1)
        coeffs[0] = 1
        coeffs[self.n] = 1
        return coeffs

    def _check_compatibility(self, other: "PolynomialRing"):
        if self.q != other.q:
            raise ValueError("Polynomials must have same q.")
        if self.n != other.n:
            raise ValueError("Polynomials must have same n.")

    def __mul__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        f = np.polynomial.Polynomial(self.x)
        g = np.polynomial.Polynomial(other.x)

        reduction_polynomial = np.polynomial.Polynomial(
            self._reduction_polynomial_coefficients()
        )

        h = f * g
        h = np.polynomial.Polynomial(np.mod(h.coef, self.q))
        r = h % reduction_polynomial

        x = r.coef
        x = np.mod(x, self.q)

        return PolynomialRing(q=self.q, n=self.n, x=x)

    def __add__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        x = self.x + other.x
        x = np.mod(x, self.q)

        return PolynomialRing(q=self.q, n=self.n, x=x)

    def __sub__(self, other: "PolynomialRing") -> "PolynomialRing":
        self._check_compatibility(other)

        x = self.x - other.x
        x = np.mod(x, self.q)

        return PolynomialRing(q=self.q, n=self.n, x=x)
