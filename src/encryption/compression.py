"""
See Section 4.2.1 of FIPS 203 regarding compression.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import math
from fractions import Fraction

from encryption.byte_encoding import count_bits
from math_.rounding import round_break_ties_upward


def compress(x: int, d: int, q: int) -> int:
    q_bits = count_bits(q)
    if d >= q_bits:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    p = 1 << d
    return round_break_ties_upward(Fraction(p, q) * x) % p


def decompress(y: int, d: int, q: int) -> int:
    q_bits = count_bits(q)
    if d >= q_bits:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    p = int(math.pow(2, d))
    return round_break_ties_upward(Fraction(q, p) * y)
