from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import json
import os

import data_loader as io_utils


# ======================================================
# CONFIGURATION
# ======================================================
@dataclass
class FeatureConfig:
    fps: float = 50.0
    smooth_window: int = 7
    local_window: int = 5   # fenêtre pour features locales


# ======================================================
# OUTILS LOCAUX
# ======================================================
def local_energy(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Énergie locale robuste (moyenne quadratique sur fenêtre glissante).
    """
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        i0 = max(0, i - w)
        i1 = min(len(arr), i + w + 1)
        out[i] = np.nanmean(arr[i0:i1] ** 2)
    return out


# ======================================================
# CINÉMATIQUE PURE (SANS LABELS)
# ======================================================
def compute_kinematics(
    frames: List[int],
    xs: np.ndarray,
    ys: np.ndarray,
    cfg: FeatureConfig
) -> Dict[str, np.ndarray]:
    """
    Cinématique image robuste pour unsupervised.
    Aucune logique hit/bounce ici.
    """

    # Dérivées (NaN-safe)
    vx = np.gradient(xs)
    vy = np.gradient(ys)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    speed = np.hypot(vx, vy)
    acc_mag = np.hypot(ax, ay)

    # Direction et changements de direction
    angles = np.unwrap(np.arctan2(vy, vx))
    turn_rate = np.abs(np.gradient(angles))

    # Sol image (approx robuste)
    ground_y = np.nanpercentile(ys, 99)

    # Distances et ruptures
    dist_to_ground = ys - ground_y
    speed_diff = np.abs(np.gradient(speed))
    vertical_shock = np.abs(np.gradient(vy))

    # Énergie locale (contexte)
    speed_energy = local_energy(speed, cfg.local_window)
    vy_energy = local_energy(vy, cfg.local_window)

    return dict(
        xs=xs,
        ys=ys,
        vx=vx,
        vy=vy,
        ax=ax,
        ay=ay,
        speed=speed,
        acc_mag=acc_mag,
        turn_rate=turn_rate,
        dist_to_ground=dist_to_ground,
        speed_diff=speed_diff,
        vertical_shock=vertical_shock,
        speed_energy=speed_energy,
        vy_energy=vy_energy,
        ground_y=np.array([ground_y], dtype=float),
    )


# ======================================================
# MATRICE DE FEATURES
# ======================================================
def build_feature_matrix(
    frames: List[int],
    xs: List[float],
    ys: List[float],
    cfg: FeatureConfig
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Retourne :
      - X     : matrice (n_frames, n_features)
      - names : noms des features
      - kin   : dictionnaire complet pour debug / plots
    """

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    kin = compute_kinematics(frames, x, y, cfg)

    # 🔥 Features sélectionnées pour unsupervised
    X = np.column_stack([
        kin["speed"],
        kin["vy"],
        kin["ay"],
        kin["turn_rate"],
        kin["acc_mag"],
        kin["dist_to_ground"],
        kin["speed_diff"],
        kin["vertical_shock"],
        kin["speed_energy"],
        kin["vy_energy"],
    ])

    names = [
        "speed",
        "vy",
        "ay",
        "turn_rate",
        "acc_mag",
        "dist_to_ground",
        "speed_diff",
        "vertical_shock",
        "speed_energy",
        "vy_energy",
    ]

    return X, names, kin


# ======================================================
# HELPERS I/O (DEBUG / ANALYSE)
# ======================================================
def analyze_rally(json_path: str, cfg: FeatureConfig | None = None):
    if cfg is None:
        cfg = FeatureConfig()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames, xs, ys, vis, actions = io_utils.extract_series(data)
    X, names, kin = build_feature_matrix(frames, xs, ys, cfg)

    return frames, xs, ys, vis, actions, X, names, kin


# ======================================================
# VISUALISATIONS SIMPLES
# ======================================================
def visualize_basic(frames: List[int], ys: List[float], kin: Dict[str, np.ndarray]):
    plt.figure(figsize=(14, 4))
    plt.plot(frames, ys, lw=1.5)
    plt.gca().invert_yaxis()
    plt.title("Trajectoire verticale (Y)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(frames, kin["speed"], lw=1.5)
    plt.title("Vitesse image (px/frame)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(frames, kin["vertical_shock"], lw=1.5)
    plt.title("Choc vertical (|dVy|)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ======================================================
# MAIN (TEST LOCAL)
# ======================================================
if __name__ == "__main__":
    path = "Data hit & bounce/per_point_v2/ball_data_230.json"
    if os.path.exists(path):
        fr, x, y, vis, act, X, names, kin = analyze_rally(path)
        print("Features :", names)
        visualize_basic(fr, y, kin)
    else:
        print("Fichier JSON introuvable.")
