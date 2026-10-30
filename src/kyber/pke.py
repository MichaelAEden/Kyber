import numpy as np

from kyber.byte_encoding import byte_encode
from kyber.key_gen import random_polynomial_matrix, random_polynomial_vector
from kyber.primitives import hash_g
from math_.matrix import Matrix
from math_.modulo import modular_rounding
from math_.ntt import PolynomialNTT
from math_.polynomial_ring import Polynomial

KyberCiphertext = tuple[Matrix[Polynomial], Matrix[Polynomial]]


class KyberPKE:
    """
    Implements Kyber public-key encryption (PKE) scheme, without ciphertext compression.

    See Section 5.1 of FIPS 203.
    https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    """

    def __init__(
        self,
        n: int,
        q: int,
        k: int,
        nu1: int,
        nu2: int,
        rho: bytes,
        A: Matrix[PolynomialNTT],
        s: Matrix[PolynomialNTT],
        e: Matrix[PolynomialNTT],
    ):
        """
        :param n: Degree of reduction polynomial.
        :param q: Prime modulus.
        :param k: Dimensions of lattice matrix.
        :param nu1: Coefficient bound for s, e, r.
        :param nu2: Coefficient bound for e1, e2.
        """
        # Parameters
        self.n = n
        self.q = q
        self.k = k
        self.nu1 = nu1
        self.nu2 = nu2
        self.rho = rho

        # Encryption, decryption keys.
        self.A = A
        self.s = s
        self.t = self.A * self.s + e

    @classmethod
    def from_seed(
        cls,
        n: int,
        q: int,
        k: int,
        nu1: int,
        nu2: int,
        d: bytes,
    ):
        rho, sigma = hash_g(d + byte_encode(k, d=8))

        # Generate lattice matrix, A.
        A = random_polynomial_matrix(q=q, n=n, k=k, seed=rho)

        # Generate secret vector, s.
        s = random_polynomial_vector(
            q=q, n=n, k=k, nu=nu1, seed_int=0, seed_bytes=sigma
        )

        # Generate error vector, e.
        e = random_polynomial_vector(
            q=q, n=n, k=k, nu=nu1, seed_int=k, seed_bytes=sigma
        )

        return cls(
            n=n,
            q=q,
            k=k,
            nu1=nu1,
            nu2=nu2,
            rho=rho,
            A=A,
            s=s,
            e=e,
        )

    @property
    def encryption_key(self) -> bytes:
        coefs = np.concatenate([p.coef for p in self.t])
        return byte_encode(coefs, d=12) + self.rho

    @property
    def decryption_key(self) -> bytes:
        coefs = np.concatenate([p.coef for p in self.s])
        return byte_encode(coefs, d=12)

    def encrypt(
        self,
        m: np.ndarray,
        r: Matrix[Polynomial] = None,
        e1: Matrix[Polynomial] = None,
        e2: Matrix[Polynomial] = None,
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
                Polynomial(q=self.q, n=self.n, coef=np.ceil(self.q / 2) * m)
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
