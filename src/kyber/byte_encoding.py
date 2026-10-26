"""
See Section 4.2.1 of FIPS 203 regarding byte encoding and decoding.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import math

import numpy as np


def count_bits(x: int) -> int:
    return math.ceil(math.log(x, 2))


def _bits_to_bytes(bit_array: np.ndarray) -> bytes:
    n, r = divmod(len(bit_array), 8)
    if r != 0:
        raise ValueError("Bit array length must be divisible by 8.")

    data = b""
    for i in range(n):
        as_int = sum(int(bit_array[i * 8 + j]) << j for j in range(8))
        data += as_int.to_bytes(length=1, byteorder="little")

    return data


def _bytes_to_bits(data: bytes) -> np.ndarray:
    n = len(data)
    bit_array = np.zeros(n * 8, dtype=bool)
    for i, a in enumerate(data):
        for j in range(8):
            bit_array[i * 8 + j] = a % 2
            a = (a - bit_array[i * 8 + j]) // 2

    return bit_array


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

    n = len(data)
    bit_array = np.zeros(d * n, dtype=bool)
    for i, a in enumerate(data.tolist()):
        for j in range(d):
            bit_array[i * d + j] = a % 2
            a = (a - bit_array[i * d + j]) // 2

    return _bits_to_bytes(bit_array)


def byte_decode(data: bytes, d: int, q: int) -> np.ndarray:
    q_bits = count_bits(q)

    if d < q_bits:
        max_value = 1 << d
    elif d == q_bits:
        max_value = q
    else:
        raise ValueError(f"d must be less than {q_bits} bits for modular base: {q}")

    n, r = divmod(len(data) * 8, d)
    if r != 0:
        raise ValueError(f"Length of data must be divisible by: {d}")

    bit_array = _bytes_to_bits(data)
    ints = []
    for i in range(n):
        ints.append(sum(int(bit_array[i * d + j]) << j for j in range(d)))
    ints = np.array(ints)

    if not np.all(ints < max_value):
        raise ValueError(f"data must be less than: {max_value}")

    return ints
