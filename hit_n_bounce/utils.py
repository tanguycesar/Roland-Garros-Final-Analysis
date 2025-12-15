from __future__ import annotations

import numpy as np
from typing import Tuple


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs (keeps leading/trailing NaNs as nearest)."""
    x = x.astype(float)
    n = len(x)
    if n == 0:
        return x
    isn = np.isnan(x)
    if not isn.any():
        return x
    idx = np.arange(n)
    good = ~isn
    if good.sum() == 0:
        return np.zeros_like(x)
    # fill edges
    first = idx[good][0]
    last = idx[good][-1]
    x[:first] = x[first]
    x[last+1:] = x[last]
    # interpolate middle
    x[isn] = np.interp(idx[isn], idx[good], x[good])
    return x


def robust_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def unit_vector(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    sp = np.sqrt(vx * vx + vy * vy) + eps
    return vx / sp, vy / sp
