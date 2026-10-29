from typing import Callable

import numpy as np

from math_.matrix import Matrix
from math_.polynomial_ring import Polynomial


def _random_polynomial(
    q: int, n: int, max_coef: int, rng: np.random.Generator
) -> Polynomial:
    coef = rng.integers(low=0, high=max_coef, size=n)
    return Polynomial(q=q, n=n, coef=coef)


def _random_matrix(row_count: int, col_count: int, fn: Callable) -> Matrix:
    return Matrix(
        tuple(tuple(fn() for _ in range(col_count)) for _ in range(row_count))
    )


def random_polynomial_matrix(
    q: int,
    n: int,
    max_coef: int,
    row_count: int,
    col_count: int,
    rng: np.random.Generator,
) -> Matrix[Polynomial]:
    return _random_matrix(
        row_count=row_count,
        col_count=col_count,
        fn=lambda: _random_polynomial(q, n, max_coef, rng),
    )
