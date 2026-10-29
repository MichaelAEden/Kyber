"""
See Section 4.3 of FIPS 203 regarding the Number-Theoretic Transform (NTT).
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import numpy as np

from kyber.byte_encoding import bit_reverse


def compute_zeta_values():
    # TODO: replace with Appendix A or cache.
    q = 3329
    n = 128
    zeta = 17
    powers = bit_reverse(np.arange(n), d=7)
    return np.array([pow(zeta, int(p), q) for p in powers])


def ntt(f: np.ndarray) -> np.ndarray:
    q = 3329

    f = np.copy(f)
    zeta_values = compute_zeta_values()

    i = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = zeta_values[i]
            i += 1
            j = start
            while j < start + length:
                t = (zeta * f[j + length]) % q
                f[j + length] = (f[j] - t) % q
                f[j] = (f[j] + t) % q
                j += 1
            start = start + 2 * length
        length = length // 2
    return f


def inverse_ntt(f: np.ndarray) -> np.ndarray:
    q = 3329

    f = np.copy(f)
    zeta_values = compute_zeta_values()

    i = 127
    length = 2
    while length <= 128:
        start = 0
        while start < 256:
            zeta = zeta_values[i]
            i -= 1
            j = start
            while j < start + length:
                t = f[j]
                f[j] = (t + f[j + length]) % q
                f[j + length] = (zeta * (f[j + length] - t)) % q
                j += 1
            start = start + 2 * length
        length = length * 2
    f = (f * 3303) % q
    return f
