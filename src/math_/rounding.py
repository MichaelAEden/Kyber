from fractions import Fraction

import numpy as np


def round_break_ties_upward(f: Fraction) -> int:
    p, q = f.as_integer_ratio()

    if p < 0:
        raise ValueError("Value must be positive.")

    n, r = p // q
    if r * 2 < q:
        return n
    else:
        return n + 1


def modular_rounding(x: np.ndarray, q: int) -> np.ndarray:
    """
    Round x to 0 if x < q/4 or x > 3q/4, round x to 1 otherwise.
    """
    x = np.mod(x, q)
    return ((x > q // 4) & (x <= 3 * q // 4)).astype(int)
