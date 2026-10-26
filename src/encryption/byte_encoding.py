"""
See Section 4.2.1 of FIPS 203 regarding byte encoding and decoding.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import math

import numpy as np


def count_bits(x: int) -> int:
    return math.ceil(math.log(x, 2))


def byte_encode(data: np.ndarray, d: int, q: int) -> bytes:
    q_bits = count_bits(q)

    if d < q_bits:
        max_value = 1 << d
    elif d == q_bits:
        max_value = q
    else:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    if not np.all(data < max_value):
        raise ValueError(f"data must be less than: {max_value}")

    b = b""
    for value in data:
        b += value.to_bytes(length=d, byteorder="little", signed=False)

    return b


def byte_decode(data: bytes, d: int, q: int) -> np.ndarray:
    q_bits = count_bits(q)

    if d < q_bits:
        max_value = 1 << d
    elif d == q_bits:
        max_value = q
    else:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    n, r = len(data) // d
    if r != 0:
        raise ValueError(f"Length of data must be divisible by: {d}")

    ints = []
    for i in range(n):
        ints.append(
            int.from_bytes(data[i * d : (i + 1) * d], byteorder="little", signed=False)
        )
    ints = np.array(ints)

    if not np.all(ints < max_value):
        raise ValueError(f"data must be less than: {max_value}")

    return ints
