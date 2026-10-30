"""
See Section 4.2.1 of FIPS 203 regarding compression.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import numpy as np

from kyber.byte_encoding import count_bits


def round_fraction(numerator: np.ndarray, denominator: int) -> np.ndarray:
    """
    Round fraction to the nearest integer without the use of floating point
    arithmetic. Ties are broken upwards.
    """
    if np.any(numerator < 0):
        raise ValueError("Numerator must be non-negative.")
    if denominator <= 0:
        raise ValueError("Denominator must be positive.")

    n, r = np.divmod(numerator, denominator)
    round_up_mask = r * 2 >= denominator

    return n + round_up_mask


def compress(x: np.ndarray, d: int, q: int) -> np.ndarray:
    q_bits = count_bits(q)
    if d >= q_bits:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    p = 1 << d
    return np.mod(round_fraction(numerator=p * x, denominator=q), p)


def decompress(y: np.ndarray, d: int, q: int) -> np.ndarray:
    q_bits = count_bits(q)
    if d >= q_bits:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    p = 1 << d
    return round_fraction(numerator=q * y, denominator=p)
