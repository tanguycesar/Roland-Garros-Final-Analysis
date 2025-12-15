from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from .features import FeatureConfig, compute_kinematics


def unsupervised_hit_bounce_detection(ball_data: Dict[str, Any], cfg: FeatureConfig | None = None) -> Dict[str, Any]:
    """
    Physics-inspired heuristics:
      - bounce: occurs near "ground" (high y), with vy sign change (+ -> -) and high |ay|
      - hit: strong direction change / speed change away from ground
    Adds pred_action per frame.
    """
    if cfg is None:
        cfg = FeatureConfig()

    frames = sorted([int(k) for k in ball_data.keys()])
    xs = [float(ball_data[str(fr)].get("x", float("nan"))) for fr in frames]
    ys = [float(ball_data[str(fr)].get("y", float("nan"))) for fr in frames]
    vis = [bool(ball_data[str(fr)].get("visible", True)) for fr in frames]

    kin = compute_kinematics(frames, xs, ys, vis, cfg)
    y = kin["ys"]; vy = kin["vy"]; ay = kin["ay"]; speed = kin["speed"]; turn = kin["turn_rate"]; ground = kin["ground_y"]

    n = len(frames)
    pred = np.array(["air"] * n, dtype=object)

    # thresholds (adaptive)
    # near ground: within 12% of y range from ground
    yr = np.nanmax(y) - np.nanmin(y)
    yr = yr if np.isfinite(yr) and yr > 1 else 200.0
    near_ground = y > (ground - 0.12 * yr)

    # bounce candidates: sign change in vy (+ -> -) with acceleration spike
    vy_prev = np.r_[vy[0], vy[:-1]]
    sign_flip = (vy_prev > 0) & (vy < 0)
    ay_thr = np.nanpercentile(np.abs(ay), 90) if n > 10 else np.nanmax(np.abs(ay))
    bounce_cand = near_ground & sign_flip & (np.abs(ay) >= ay_thr)

    # choose local maxima of bounce score to avoid clusters
    bounce_score = np.abs(ay) + 0.5 * speed
    bounce_idx = np.where(bounce_cand)[0]
    # non-maximum suppression within +/-2 frames
    picked = []
    for i in bounce_idx:
        if any(abs(i - j) <= 2 for j in picked):
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        j = lo + int(np.argmax(bounce_score[lo:hi]))
        if j not in picked:
            picked.append(j)
    for j in picked:
        pred[j] = "bounce"

    # hit candidates: strong direction change or speed jump, but not near ground
    turn_thr = np.nanpercentile(turn, 92) if n > 10 else np.nanmax(turn)
    speed_d = np.abs(np.r_[0.0, np.diff(speed)])
    speed_thr = np.nanpercentile(speed_d, 92) if n > 10 else np.nanmax(speed_d)
    hit_cand = (~near_ground) & ((turn >= turn_thr) | (speed_d >= speed_thr))

    # exclude close to bounce frames
    for j in picked:
        lo, hi = max(0, j - 3), min(n, j + 4)
        hit_cand[lo:hi] = False

    # pick sparse hits as well
    hit_score = turn + 0.2 * speed_d
    hit_idx = np.where(hit_cand)[0]
    picked_h = []
    for i in hit_idx:
        if any(abs(i - j) <= 2 for j in picked_h):
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        j = lo + int(np.argmax(hit_score[lo:hi]))
        if j not in picked_h:
            picked_h.append(j)
    for j in picked_h:
        pred[j] = "hit"

    # write back
    out = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])
        # if not visible, default to air
        if not bool(d.get("visible", True)):
            d["pred_action"] = "air"
        else:
            d["pred_action"] = str(pred[i])
        out[str(fr)] = d
    return out
