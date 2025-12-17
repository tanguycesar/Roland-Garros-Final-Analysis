from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json
import os

# ======================================================
# CONFIGURATION
# ======================================================
@dataclass
class FeatureConfig:
    fps: float = 25.0
    smooth_window: int = 7

# ======================================================
# 1. CAMÉRA & CALIBRATION (inchangé)
# ======================================================
class CameraModel:
    def __init__(self):
        if os.path.exists("Camera_Params_Distorted.npz"):
            data = np.load("Camera_Params_Distorted.npz")
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            self.rvec = data["rvec"]
            self.tvec = data["tvec"]
        else:
            print("⚠️ Calibration caméra absente → mode dummy")
            self.mtx = np.eye(3)
            self.dist = np.zeros((5, 1))
            self.rvec = np.zeros((3, 1))
            self.tvec = np.zeros((3, 1))

        rotM, _ = cv2.Rodrigues(self.rvec)
        self.cam_pos = -rotM.T @ self.tvec
        self.inv_rot = rotM.T
        self.inv_mtx = np.linalg.inv(self.mtx)

    def pixel_to_ground(self, u, v):
        if np.isnan(u) or np.isnan(v):
            return None
        pts = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.mtx, self.dist, P=self.mtx)[0][0]
        ray = self.inv_rot @ (self.inv_mtx @ np.array([*und, 1.0]))
        if abs(ray[2]) < 1e-6:
            return None
        l = -self.cam_pos[2] / ray[2]
        if l < 0:
            return None
        return (self.cam_pos + l * ray.reshape(3, 1)).flatten()

# ======================================================
# 2. NETTOYAGE NEUTRE (LABEL-FREE)
# ======================================================
def clean_velocity_outliers(x, y, max_jump_px=90.0):
    for _ in range(3):
        valid = np.where(~np.isnan(x))[0]
        if len(valid) < 2:
            break
        d = np.hypot(np.diff(x[valid]), np.diff(y[valid]))
        bad = np.where(d > max_jump_px)[0]
        if len(bad) == 0:
            break
        x[valid[bad + 1]] = np.nan
        y[valid[bad + 1]] = np.nan
    return x, y

def clean_local_spikes(x, y, thr=25.0):
    valid = np.where(~np.isnan(x))[0]
    for i in range(1, len(valid) - 1):
        p, c, n = valid[i - 1], valid[i], valid[i + 1]
        pred = 0.5 * (x[p] + x[n]), 0.5 * (y[p] + y[n])
        if np.hypot(x[c] - pred[0], y[c] - pred[1]) > thr:
            x[c] = np.nan
            y[c] = np.nan
    return x, y

def remove_short_segments(x, y, min_len=6):
    mask = ~np.isnan(x)
    diff = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < min_len:
            x[s:e] = np.nan
            y[s:e] = np.nan
    return x, y

def interpolate_small_gaps(a, max_gap=5):
    idx = np.arange(len(a))
    valid = ~np.isnan(a)
    if valid.sum() < 2:
        return a
    interp = np.interp(idx, idx[valid], a[valid])
    gaps = np.where(np.diff(np.where(valid)[0]) > max_gap)[0]
    for g in gaps:
        i0 = np.where(valid)[0][g] + 1
        i1 = np.where(valid)[0][g + 1]
        interp[i0:i1] = np.nan
    return interp

def smooth_segments(x, y, window=7):
    mask = ~np.isnan(x)
    diff = np.diff(np.r_[0, mask.astype(int), 0])
    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        if e - s >= window:
            w = window if window % 2 else window - 1
            x[s:e] = savgol_filter(x[s:e], w, 2)
            y[s:e] = savgol_filter(y[s:e], w, 2)
    return x, y

def process_trajectory(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    x, y = clean_velocity_outliers(x, y)
    x, y = clean_local_spikes(x, y)
    x, y = remove_short_segments(x, y)
    x = interpolate_small_gaps(x)
    y = interpolate_small_gaps(y)
    x, y = smooth_segments(x, y)
    return x, y

# ======================================================
# 3. CINÉMATIQUE PURE (SANS LABELS)
# ======================================================
def compute_kinematics(frames, xs, ys, cfg: FeatureConfig):
    vx = np.gradient(xs)
    vy = np.gradient(ys)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    speed = np.hypot(vx, vy)
    angles = np.unwrap(np.arctan2(vy, vx))
    turn_rate = np.abs(np.gradient(angles))

    ground_y = np.nanpercentile(ys, 99)

    return dict(
        xs=xs, ys=ys,
        vx=vx, vy=vy,
        ax=ax, ay=ay,
        speed=speed,
        turn_rate=turn_rate,
        ground_y=ground_y
    )

# ======================================================
# 4. ANALYSE PRINCIPALE (SANS HIT/BOUNCE)
# ======================================================
def analyze_rally(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = sorted(int(k) for k in data.keys())
    xs = [float(data[str(f)].get("x")) if data[str(f)].get("x") is not None else np.nan for f in frames]
    ys = [float(data[str(f)].get("y")) if data[str(f)].get("y") is not None else np.nan for f in frames]

    xs, ys = process_trajectory(xs, ys)
    kin = compute_kinematics(frames, xs, ys, FeatureConfig())

    return frames, xs, ys, kin

# ======================================================
# 5. VISUALISATION SIMPLE
# ======================================================
def visualize(frames, xs, ys, kin):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(frames, ys, lw=1.5)
    ax1.invert_yaxis()
    ax1.set_title("Trajectoire verticale (Y)")
    ax1.grid(True)

    ax2.plot(frames, kin["speed"], lw=1.5)
    ax2.set_title("Vitesse image (px/frame)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ======================================================
if __name__ == "__main__":
    path = "Data hit & bounce/per_point_v2/ball_data_230.json"
    if os.path.exists(path):
        fr, x, y, kin = analyze_rally(path)
        visualize(fr, x, y, kin)
    else:
        print("Fichier JSON introuvable.")
