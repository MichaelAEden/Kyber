import numpy as np

from math_.modulo import modular_rounding


class LindnerPeikert:
    """
    Implements Lindner-Peikert public-key encryption scheme.

    See Section 4.2 in the following document:
    https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
    """

    def __init__(self, n: int, q: int, B: int, A: np.ndarray = None, s: np.ndarray = None, e: np.ndarray = None, seed: int = None):
        self.rng = np.random.default_rng(seed)

        # Parameters: n, B, q
        self.n = n
        self.B = B
        self.q = q

        # Public key: (A, s)
        self.A = A if A is not None else self.rng.integers(low=0, high=q, size=(n, n))
        self.s = s if s is not None else self.rng.integers(low=-B, high=B, size=(n, 1))

        # Private key: b
        e = e if e is not None else self.rng.integers(low=-B, high=B, size=(n, 1))
        b = np.matmul(self.A, self.s) + e
        b = np.mod(b, q)
        self.b = b

    def encrypt(self, x: np.ndarray, r: np.ndarray = None, e1: np.ndarray = None, e2: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        r = r if r is not None else self.rng.integers(low=-self.B, high=self.B, size=(self.n, 1))
        e1 = e1 if e1 is not None else self.rng.integers(low=-self.B, high=self.B, size=(self.n, 1))
        e2 = e2 if e2 is not None else self.rng.integers(low=-self.B, high=self.B, size=(1, 1))

        u = np.matmul(self.A.transpose(), r) + e1
        u = np.mod(u, self.q)
        # Use ceil instead of round such that ties are broken upwards.
        v = np.matmul(self.b.transpose(), r) + e2 + np.ceil(self.q / 2) * x
        v = np.mod(v, self.q)

        return u, v

    def decrypt(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        x = modular_rounding(v - np.matmul(self.s.transpose(), u), q=self.q)
        return x.flatten()
