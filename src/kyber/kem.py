from kyber.pke import KyberPKE
from kyber.primitives import hash_h, hash_g


class KyberKEM:
    """
    Implements Kyber key encapsulation mechanism (KEM).

    See Section 6 of FIPS 203.
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
        seed: bytes,
    ):
        """
        :param n: Degree of reduction polynomial.
        :param q: Prime modulus.
        :param k: Dimensions of lattice matrix.
        :param nu1: Coefficient bound for s, e, r.
        :param nu2: Coefficient bound for e1, e2.
        :param du: Coefficient for compression of u in ciphertext.
        :param dv: Coefficient for compression of v in ciphertext.
        :param seed: Seed for RNG.
        """
        if n != 256 or q != 3329:
            raise ValueError("Only n = 256, q = 3329 is supported.")

        self.kyber_pke = KyberPKE.from_seed(
            n=n,
            q=q,
            k=k,
            nu1=nu1,
            nu2=nu2,
            du=du,
            dv=dv,
            seed=seed,
        )

    @classmethod
    def ml_kem_512(
        cls,
        seed: bytes,
    ) -> "KyberKEM":
        return cls(
            n=256,
            q=3329,
            k=2,
            nu1=3,
            nu2=2,
            du=10,
            dv=4,
            seed=seed,
        )

    @classmethod
    def ml_kem_768(
        cls,
        seed: bytes,
    ) -> "KyberKEM":
        return cls(
            n=256,
            q=3329,
            k=3,
            nu1=2,
            nu2=2,
            du=10,
            dv=4,
            seed=seed,
        )

    @classmethod
    def ml_kem_1024(
        cls,
        seed: bytes,
    ) -> "KyberKEM":
        return cls(
            n=256,
            q=3329,
            k=4,
            nu1=2,
            nu2=2,
            du=11,
            dv=5,
            seed=seed,
        )

    def encapsulate(self, m: bytes) -> tuple[bytes, bytes]:
        h = hash_h(self.kyber_pke.ek)
        K, r = hash_g(m + h)

        c = self.kyber_pke.encrypt(m, r)

        return K, c

    def decapsulate(self, c: bytes) -> bytes:
        m = self.kyber_pke.decrypt(c)

        h = hash_h(self.kyber_pke.ek)
        K, r = hash_g(m + h)

        c_prime = self.kyber_pke.encrypt(m, r)

        if c != c_prime:
            raise ValueError("Decapsulation failed.")

        return K
