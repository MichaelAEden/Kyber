"""
See Section 4.1 of FIPS 203 regarding choices of hash functions.
https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
"""

import hashlib
from typing import Callable

from Crypto.Hash import SHAKE128


def _hash_wrapper(
    data: bytes,
    hash_fn: Callable,
    output_bytes: int = None,
) -> bytes:
    result = hash_fn(data)
    if output_bytes is not None:
        return result.digest(length=output_bytes)
    else:
        return result.digest()


def pseudorandom_function(s: bytes, b: bytes, nu: int) -> bytes:
    if nu != 2 and nu != 3:
        raise ValueError("nu must be 2 or 3.")
    if len(s) != 32 or len(b) != 1:
        raise ValueError("s, b must be 32 bytes, 1 byte respectively.")

    return _hash_wrapper(
        data=(s + b),
        hash_fn=hashlib.shake_256,
        output_bytes=64 * nu,
    )


def hash_g(c: bytes) -> tuple[bytes, bytes]:
    digest = _hash_wrapper(
        data=c,
        hash_fn=hashlib.sha3_512,
    )
    return digest[:32], digest[32:]


def hash_h(s: bytes) -> bytes:
    return _hash_wrapper(
        data=s,
        hash_fn=hashlib.sha3_256,
    )


def hash_j(s: bytes) -> bytes:
    return _hash_wrapper(
        data=s,
        hash_fn=hashlib.shake_256,
        output_bytes=32,
    )


class ExtendableOutputFunction:

    def __init__(self, seed: bytes):
        self._shake128 = SHAKE128.new()
        self._shake128.update(seed)

    def squeeze(self, length: int) -> bytes:
        return self._shake128.read(length)
