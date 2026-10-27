import numpy as np


def modular_rounding(x: np.ndarray, q: int) -> np.ndarray:
    """
    Round x to 0 if x < q/4 or x > 3q/4, round x to 1 otherwise.
    """
    x = np.mod(x, q)
    return ((x > q // 4) & (x <= 3 * q // 4)).astype(int)
