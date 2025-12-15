from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.signal import savgol_filter

from .utils import interpolate_nans, robust_percentile, unit_vector


@dataclass(frozen=True)
class FeatureConfig:
    fps: float = 25.0
    sg_window: int = 11          # odd
    sg_poly: int = 2
    ctx: int = 2                 # context frames on each side (total 2*ctx+1)


def _savgol_safe(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = len(x)
    w = window
    if n < 3:
        return x.copy()
    if w >= n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return x.copy()
    if poly >= w:
        poly = max(1, w - 2)
    return savgol_filter(x, window_length=w, polyorder=poly, mode="interp")


def compute_kinematics(frames: List[int], xs: List[float], ys: List[float], vis: List[bool], cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    """Compute smoothed x/y, velocities, accelerations, and some derived signals."""
    t = np.asarray(frames, dtype=float)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    v = np.asarray(vis, dtype=bool)

    # mark non-visible as NaN to avoid weird derivatives
    x = x.copy(); y = y.copy()
    x[~v] = np.nan
    y[~v] = np.nan

    x = interpolate_nans(x)
    y = interpolate_nans(y)

    xs = _savgol_safe(x, cfg.sg_window, cfg.sg_poly)
    ys = _savgol_safe(y, cfg.sg_window, cfg.sg_poly)

    # dt in seconds per frame (assume constant fps if frames increments are 1, otherwise use actual frame ids)
    # Use frame index spacing to be safe:
    dt_frame = np.gradient(t)  # in frames
    dt = dt_frame / cfg.fps    # in seconds

    vx = np.gradient(xs) / dt_frame
    vy = np.gradient(ys) / dt_frame
    ax = np.gradient(vx) / dt_frame
    ay = np.gradient(vy) / dt_frame

    speed = np.sqrt(vx * vx + vy * vy)
    ux, uy = unit_vector(vx, vy)
    # curvature proxy: change of direction
    dux = np.gradient(ux) / dt_frame
    duy = np.gradient(uy) / dt_frame
    turn_rate = np.sqrt(dux * dux + duy * duy)

    ground_y = robust_percentile(ys, 95.0)  # in image coords, ground tends to be high y
    return dict(
        t=t, xs=xs, ys=ys, vx=vx, vy=vy, ax=ax, ay=ay,
        speed=speed, turn_rate=turn_rate, ground_y=np.full_like(ys, ground_y)
    )


def _stack_context(feat: np.ndarray, ctx: int) -> np.ndarray:
    """Return [n, 2*ctx+1] context window for a 1D feature using edge padding."""
    n = feat.shape[0]
    if ctx <= 0:
        return feat.reshape(n, 1)
    pad = np.pad(feat, (ctx, ctx), mode="edge")
    cols = []
    for k in range(2 * ctx + 1):
        cols.append(pad[k:k + n])
    return np.stack(cols, axis=1)


def make_frame_features(kin: Dict[str, np.ndarray], cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    """Create per-frame features with temporal context (suitable for non-sequence sklearn models)."""
    xs, ys = kin["xs"], kin["ys"]
    vx, vy = kin["vx"], kin["vy"]
    ax, ay = kin["ax"], kin["ay"]
    speed = kin["speed"]
    turn_rate = kin["turn_rate"]
    ground_y = kin["ground_y"]

    # distance to (estimated) ground line
    y_to_ground = ground_y - ys

    base = {
        "xs": xs,
        "ys": ys,
        "vx": vx, "vy": vy,
        "ax": ax, "ay": ay,
        "speed": speed,
        "turn_rate": turn_rate,
        "y_to_ground": y_to_ground,
        "abs_ay": np.abs(ay),
        "abs_ax": np.abs(ax),
    }

    feats = []
    names = []
    for name, arr in base.items():
        ctx_arr = _stack_context(arr, cfg.ctx)  # [n, w]
        w = ctx_arr.shape[1]
        for j in range(w):
            feats.append(ctx_arr[:, j])
            names.append(f"{name}_t{j - cfg.ctx:+d}")
    X = np.stack(feats, axis=1)
    return X, names
