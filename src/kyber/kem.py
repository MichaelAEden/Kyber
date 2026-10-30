import hashlib
from typing import Callable

import numpy as np

from kyber.pke import KyberPKE


def hash_(hash_fn: Callable) -> Callable[[any], bytes]:
    def _hash(x) -> bytes:
        return hash_fn(hash(x).to_bytes(length=32, signed=True)).digest()

    return _hash


class KyberKEM:
    """
    Implements Kyber key encapsulation mechanism (KEM).

    See Section 6.4 in the following document:
    https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
    """

    def __init__(
        self,
        n: int,
        q: int,
        k: int,
        nu1: int,
        nu2: int,
        seed: bytes,
    ):
        """
        :param n: Degree of reduction polynomial.
        :param q: Prime modulus.
        :param k: Dimensions of lattice matrix.
        :param nu1: Coefficient bound for s, e, r.
        :param nu2: Coefficient bound for e1, e2.
        :param seed: Seed for RNG.
        """
        if n != 256:
            raise ValueError("Only n = 256 is supported.")

        self.hash_fn_g = hash_(hashlib.sha3_512)
        self.hash_fn_h = hash_(hashlib.sha3_256)
        self.hash_fn_j = hash_(hashlib.sha3_256)

        self.kyber_pke = KyberPKE.from_seed(
            n=n,
            q=q,
            k=k,
            nu1=nu1,
            nu2=nu2,
            d=seed,
        )
        rng = np.random.default_rng(seed=2026)

        self.z = rng.integers(low=0, high=1, endpoint=True, size=(256,))

    def encapsulate(self, seed: int = None) -> tuple[bytes, bytes]:
        rng = np.random.default_rng(seed)

        m = rng.integers(low=0, high=1, endpoint=True, size=(256,)).astype(bool)

        h = self.hash_fn_h(self.kyber_pke.encryption_key)
        K_R = self.hash_fn_g((m.tobytes(), h))
        K, R = K_R[:128], K_R[128:]

        c = self.kyber_pke._encrypt(m=m, seed=int.from_bytes(R))

        return K, c

    def decapsulate(self, c: bytes) -> bytes:
        m = self.kyber_pke.decrypt(c)

        h = self.hash_fn_h(self.kyber_pke.encryption_key)
        K_R = self.hash_fn_g((m.tobytes(), h))
        K, R = K_R[:128], K_R[128:]

        c_prime = self.kyber_pke._encrypt(m=m, seed=int.from_bytes(R))

        if c != c_prime:
            return self.hash_fn_j((self.z, c))
        else:
            return K
