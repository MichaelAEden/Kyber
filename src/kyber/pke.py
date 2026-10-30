import numpy as np

from kyber.byte_encoding import byte_encode, byte_decode, bit_decode, bytes_to_bits
from kyber.compression import decompress, compress
from kyber.key_gen import random_polynomial_matrix_ntt, random_polynomial_vector
from kyber.primitives import hash_g
from math_.matrix import Matrix
from math_.ntt import PolynomialNTT, inverse_ntt, ntt
from math_.polynomial_ring import Polynomial


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
        du: int,
        dv: int,
        rho: bytes,
        A_ntt: Matrix[PolynomialNTT],
        s_ntt: Matrix[PolynomialNTT],
        e_ntt: Matrix[PolynomialNTT],
    ):
        """
        :param n: Degree of reduction polynomial.
        :param q: Prime modulus.
        :param k: Dimensions of lattice matrix.
        :param nu1: Coefficient bound for s, e, r.
        :param nu2: Coefficient bound for e1, e2.
        :param du: Coefficient for compression of u in ciphertext.
        :param dv: Coefficient for compression of v in ciphertext.
        :param rho: Seed for randomness.
        """
        # Parameters
        self.n = n
        self.q = q
        self.k = k
        self.nu1 = nu1
        self.nu2 = nu2
        self.du = du
        self.dv = dv
        self.rho = rho

        # Encryption, decryption keys.
        self.A_ntt = A_ntt
        self.s_ntt = s_ntt
        self.t_ntt = self.A_ntt * self.s_ntt + e_ntt

    @classmethod
    def from_seed(
        cls,
        n: int,
        q: int,
        k: int,
        nu1: int,
        nu2: int,
        du: int,
        dv: int,
        seed: bytes,
    ):
        rho, sigma = hash_g(seed + byte_encode(k, d=8))

        # Generate lattice matrix, A.
        A_ntt = random_polynomial_matrix_ntt(q=q, n=n, k=k, seed=rho)

        # Generate secret vector, s.
        s_ntt = random_polynomial_vector(
            q=q, n=n, k=k, nu=nu1, seed_int=0, seed_bytes=sigma
        ).transform(ntt)

        # Generate error vector, e.
        e_ntt = random_polynomial_vector(
            q=q, n=n, k=k, nu=nu1, seed_int=k, seed_bytes=sigma
        ).transform(ntt)

        return cls(
            n=n,
            q=q,
            k=k,
            nu1=nu1,
            nu2=nu2,
            du=du,
            dv=dv,
            rho=rho,
            A_ntt=A_ntt,
            s_ntt=s_ntt,
            e_ntt=e_ntt,
        )

    @property
    def encryption_key(self) -> bytes:
        coefs = np.concatenate([p.coef for p in self.t_ntt])
        return byte_encode(coefs, d=12) + self.rho

    @property
    def decryption_key(self) -> bytes:
        coefs = np.concatenate([p.coef for p in self.s_ntt])
        return byte_encode(coefs, d=12)

    def encrypt(self, m: bytes, seed: bytes) -> bytes:
        y = random_polynomial_vector(
            q=self.q, n=self.n, k=self.k, nu=self.nu1, seed_int=0, seed_bytes=seed
        )
        e1 = random_polynomial_vector(
            q=self.q, n=self.n, k=self.k, nu=self.nu2, seed_int=self.k, seed_bytes=seed
        )
        e2 = random_polynomial_vector(
            q=self.q, n=self.n, k=1, nu=self.nu2, seed_int=self.k * 2, seed_bytes=seed
        )
        u, v = self._encrypt(m=bytes_to_bits(m), y=y, e1=e1, e2=e2)
        c1 = byte_encode(
            compress(
                np.concatenate([p.coef for p in u]),
                d=self.du,
                q=self.q,
            ),
            d=self.du,
        )
        c2 = byte_encode(
            compress(
                np.concatenate([p.coef for p in v]),
                d=self.dv,
                q=self.q,
            ),
            d=self.dv,
        )
        return c1 + c2

    def _encrypt(
        self,
        m: np.ndarray,
        y: Matrix[Polynomial],
        e1: Matrix[Polynomial],
        e2: Matrix[Polynomial],
    ) -> tuple[Matrix[Polynomial], Matrix[Polynomial]]:
        y_ntt = y.transform(ntt)
        u = (self.A_ntt.transpose * y_ntt).transform(inverse_ntt) + e1
        mu = Matrix.scalar(
            Polynomial(
                coef=decompress(bit_decode(m, d=1), d=1, q=self.q),
                q=self.q,
                n=self.n,
            )
        )
        v = (self.t_ntt.transpose * y_ntt).transform(inverse_ntt) + e2 + mu
        return u, v

    def decrypt(
        self,
        c: bytes,
    ) -> bytes:
        i = 32 * self.du * self.k
        c1, c2 = c[:i], c[i:]

        # Obtain u, v from raw ciphertext.
        u_coef = decompress(
            byte_decode(c1, d=self.du),
            d=self.du,
            q=self.q,
        )
        u = Matrix.vector(
            Polynomial(
                coef=u_coef[i * self.n : (i + 1) * self.n],
                q=self.q,
                n=self.n,
            )
            for i in range(self.k)
        )
        v_coef = decompress(
            byte_decode(c2, d=self.dv),
            d=self.dv,
            q=self.q,
        )
        v = Matrix.scalar(
            Polynomial(
                coef=v_coef,
                q=self.q,
                n=self.n,
            )
        )

        return byte_encode(self._decrypt(u=u, v=v), d=1)

    def _decrypt(
        self,
        u: Matrix[Polynomial],
        v: Matrix[Polynomial],
    ) -> np.ndarray:
        w = v - (self.s_ntt.transpose * u.transform(ntt)).transform(inverse_ntt)
        w_coef = np.concatenate([p.coef for p in w])
        return compress(w_coef, d=1, q=self.q)
