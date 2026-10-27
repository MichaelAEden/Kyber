"""
See Section 4.2.2 of FIPS 203 regarding sampling.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import numpy as np

from kyber.primitives import ExtendableOutputFunction


def sample_ntt(seed: bytes, q: int) -> np.ndarray:
    if len(seed) != 34:
        raise ValueError("Seed must be 34 bytes.")

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
