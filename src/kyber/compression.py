"""
See Section 4.2.1 of FIPS 203 regarding compression.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

from fractions import Fraction

from kyber.byte_encoding import count_bits


def round_break_ties_upward(f: Fraction) -> int:
    p, q = f.as_integer_ratio()

    if p < 0:
        raise ValueError("Value must be positive.")

    n, r = divmod(p, q)
    if r * 2 < q:
        return n
    else:
        return n + 1


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

    p = 1 << d
    return round_break_ties_upward(Fraction(q, p) * y)
