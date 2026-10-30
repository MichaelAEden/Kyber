import numpy as np

from kyber.byte_encoding import byte_encode
from kyber.primitives import pseudorandom_function
from kyber.sampling import sample_ntt, sample_polynomial
from math_.matrix import Matrix
from math_.ntt import PolynomialNTT, ntt


def random_polynomial_vector(
    q: int, n: int, k: int, nu: int, seed_int: int, seed_bytes: bytes
) -> Matrix[PolynomialNTT]:
    v = [
        ntt(
            sample_polynomial(
                pseudorandom_function(
                    seed_bytes, byte_encode(i + seed_int, d=8), nu=nu
                ),
                nu=nu,
                q=q,
                n=n,
            )
        )
        for i in range(k)
    ]
    return Matrix.vector(v, row_vector=False)


def random_polynomial_matrix(
    q: int,
    n: int,
    k: int,
    seed: bytes,
) -> Matrix[PolynomialNTT]:
    A_ntt = np.zeros((k, k)).tolist()
    for i in range(k):
        for j in range(k):
            A_ntt[i][j] = sample_ntt(
                seed + byte_encode(np.array([j, i]), d=8), q=q, n=n
            )
    A_ntt = Matrix(A_ntt)
    return A_ntt
