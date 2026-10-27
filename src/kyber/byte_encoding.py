"""
See Section 4.2.1 of FIPS 203 regarding byte encoding and decoding.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import math

import numpy as np


def count_bits(n: int) -> int:
    return math.ceil(math.log(n, 2))


def _int_to_bits(n: int, d: int) -> np.ndarray:
    if n >= 1 << d:
        raise OverflowError(f"Cannot encode {n} into {d} bits.")

    bits = np.zeros(d, dtype=bool)
    for i in range(d):
        bits[i] = n % 2
        n = (n - bits[i]) // 2
    return bits


def _bits_to_int(bits: np.ndarray) -> int:
    return sum(int(bit) << i for (i, bit) in enumerate(bits))


def bit_encode(integers: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([_int_to_bits(n, d) for n in integers])


def bit_decode(bits: np.ndarray, d: int) -> np.ndarray:
    # In FIPS 203, this algorithm outputs integers modulo q.
    # For simplicity, the modular step was omitted here and implemented
    # downstream instead.
    # TODO: check all downstream usages.
    n, r = divmod(len(bits), d)
    if r != 0:
        raise ValueError(f"Bit array length must be divisible by {d}.")

    return np.array(
        [_bits_to_int(bits[i * d : (i + 1) * d]) for i in range(n)],
        dtype=np.uint16,
    )


def bits_to_bytes(bits: np.ndarray) -> bytes:
    integers = bit_decode(bits, d=8)
    return integers.astype(np.uint8).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    integers = np.array([data[i] for i in range(0, len(data))])
    return bit_encode(integers, d=8)
