"""
See Section 4.2.2 of FIPS 203 regarding sampling.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import numpy as np

from kyber.byte_encoding import bytes_to_bits
from kyber.primitives import ExtendableOutputFunction


def sample_ntt(seed: bytes, q: int) -> np.ndarray:
    """
    Sample coefficients from a uniform distribution to represent a polynomial
    in the NTT domain.
    """
    seed_length = 34
    if len(seed) != seed_length:
        raise ValueError(f"Seed must be {seed_length} bytes.")

    n = 256
    a = np.zeros(n, dtype=np.uint16)

    xof = ExtendableOutputFunction(seed)

    i = 0
    while i < 256:
        c = xof.squeeze(length=3)
        d1 = c[0] + 256 * (c[1] % 16)
        d2 = c[1] // 16 + 16 * c[2]

        if d1 < q:
            a[i] = d1
            i += 1

        if d2 < q and i < 256:
            a[i] = d2
            i += 1

    return a


def sample_polynomial(seed: bytes, nu: int, q: int) -> np.ndarray:
    """
    Sample polynomial coefficients from a centered binomial distribution.
    """
    if nu != 2 and nu != 3:
        raise ValueError(f"nu must be 2 or 3.")

    seed_length = 64 * nu
    if len(seed) != seed_length:
        raise ValueError(f"Seed must be {seed_length} bytes.")

    n = 256
    f = np.zeros(n)
    bits = bytes_to_bits(seed)
    for i in range(n):
        x = sum(bits[2 * i * nu + j] for j in range(nu))
        y = sum(bits[2 * i * nu + nu + j] for j in range(nu))
        f[i] = (x - y) % q

    return f
