from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
from scipy.signal import savgol_filter

# ======================================================
# Constantes
# ======================================================
DEFAULT_ACTION = "air"
DEFAULT_VISIBLE = True


# ======================================================
# Chargement et sauvegarde des JSON
# ======================================================
def load_ball_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def save_ball_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_point_files(points_dir: str | Path) -> List[Path]:
    points_dir = Path(points_dir)
    if not points_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {points_dir}")
    return sorted(points_dir.rglob("*.json"))


# ======================================================
# Nettoyage géométrique
# ======================================================
def clean_velocity_outliers(
    x: np.ndarray, y: np.ndarray, max_jump_px: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(3):
        valid = np.where(~np.isnan(x) & ~np.isnan(y))[0]
        if len(valid) < 2:
            break
        dx = np.diff(x[valid])
        dy = np.diff(y[valid])
        if len(dx) == 0:
            break
        dist = np.hypot(dx, dy)
        bad = np.where(dist > max_jump_px)[0]
        if len(bad) == 0:
            break
        x[valid[bad + 1]] = np.nan
        y[valid[bad + 1]] = np.nan
    return x, y


def clean_local_spikes(
    x: np.ndarray, y: np.ndarray, threshold_px: float = 25.0
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.where(~np.isnan(x) & ~np.isnan(y))[0]
    if len(valid) < 3:
        return x, y

    to_nan = []
    for i in range(1, len(valid) - 1):
        i0, i1, i2 = valid[i - 1], valid[i], valid[i + 1]
        px = 0.5 * (x[i0] + x[i2])
        py = 0.5 * (y[i0] + y[i2])
        if np.hypot(x[i1] - px, y[i1] - py) > threshold_px:
            to_nan.append(i1)

    if to_nan:
        x[to_nan] = np.nan
        y[to_nan] = np.nan

    return x, y


def remove_short_segments(
    x: np.ndarray, y: np.ndarray, min_len: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(mask):
        return x, y

    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if e - s < min_len:
            x[s:e] = np.nan
            y[s:e] = np.nan

    return x, y


# ======================================================
# 🔥 Détection du VRAI début du point
# ======================================================
def find_real_rally_start(
    x: np.ndarray,
    y: np.ndarray,
    min_len: int = 25,
    min_y_range: float = 120.0,
) -> int:
    """
    Vrai départ du point = segment long + vraie dynamique verticale.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if e - s < min_len:
            continue
        y_seg = y[s:e]
        if np.nanmax(y_seg) - np.nanmin(y_seg) < min_y_range:
            continue
        return s

    # fallback : plus long segment
    if len(starts) > 0:
        lengths = ends - starts
        return starts[np.argmax(lengths)]

    return 0


def crop_before_rally(x: np.ndarray, y: np.ndarray, start: int):
    x[:start] = np.nan
    y[:start] = np.nan
    return x, y


# ======================================================
# Raccord simple des segments
# ======================================================
def connect_close_segments(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 8,
    max_dist_px: float = 140.0,
):
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for i in range(len(starts) - 1):
        e1 = ends[i]
        s2 = starts[i + 1]
        gap = s2 - e1
        if gap <= 0 or gap > max_gap:
            continue

        dx = x[s2] - x[e1 - 1]
        dy = y[s2] - y[e1 - 1]
        if np.hypot(dx, dy) > max_dist_px:
            continue

        t = np.linspace(0, 1, gap + 2)
        x[e1 - 1 : s2 + 1] = x[e1 - 1] + dx * t
        y[e1 - 1 : s2 + 1] = y[e1 - 1] + dy * t

    return x, y


# ======================================================
# Interpolation locale stricte
# ======================================================
def interpolate_small_gaps_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 5,
    max_step_px: float = 70.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts_nan = np.where(diff == -1)[0]
    ends_nan = np.where(diff == 1)[0]

    for s_nan, e_nan in zip(starts_nan, ends_nan):
        gap = e_nan - s_nan
        if gap <= 0 or gap > max_gap:
            continue

        left = s_nan - 1
        right = e_nan
        if left < 0 or right >= len(x):
            continue

        dx = x[right] - x[left]
        dy = y[right] - y[left]
        if np.hypot(dx, dy) > max_step_px * (gap + 1):
            continue

        t = np.linspace(0, 1, gap + 2)
        x[left:right + 1] = x[left] + dx * t
        y[left:right + 1] = y[left] + dy * t

    return x, y


# ======================================================
# Lissage
# ======================================================
def smooth_segments(
    x: np.ndarray, y: np.ndarray, window: int = 7, poly: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        length = e - s
        if length >= window:
            w = window if window % 2 else window - 1
            try:
                x[s:e] = savgol_filter(x[s:e], w, poly)
                y[s:e] = savgol_filter(y[s:e], w, poly)
            except Exception:
                pass

    return x, y

def connect_medium_gaps(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 20,
    max_dist_px: float = 260.0,
    max_angle_deg: float = 45.0,
):
    """
    Raccorde des trous intermédiaires s'ils sont cohérents en direction.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for i in range(len(starts) - 1):
        e1 = ends[i]
        s2 = starts[i + 1]
        gap = s2 - e1

        if gap <= 0 or gap > max_gap:
            continue

        # vecteur avant
        if e1 - 3 < 0 or s2 + 3 >= len(x):
            continue

        v1 = np.array([
            x[e1 - 1] - x[e1 - 3],
            y[e1 - 1] - y[e1 - 3],
        ])
        v2 = np.array([
            x[s2 + 2] - x[s2],
            y[s2 + 2] - y[s2],
        ])

        if np.linalg.norm(v1) < 1e-3 or np.linalg.norm(v2) < 1e-3:
            continue

        # angle entre directions
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        if angle > max_angle_deg:
            continue

        # distance globale
        dx = x[s2] - x[e1 - 1]
        dy = y[s2] - y[e1 - 1]
        if np.hypot(dx, dy) > max_dist_px:
            continue

        # interpolation
        t = np.linspace(0, 1, gap + 2)
        x[e1 - 1 : s2 + 1] = x[e1 - 1] + dx * t
        y[e1 - 1 : s2 + 1] = y[e1 - 1] + dy * t

    return x, y

def connect_very_small_gaps(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 4,
    max_dist_px: float = 180.0,
):
    """
    Raccord final très permissif pour supprimer les micro-trous visuels.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for i in range(len(starts) - 1):
        e1 = ends[i]
        s2 = starts[i + 1]
        gap = s2 - e1

        if gap <= 0 or gap > max_gap:
            continue

        dx = x[s2] - x[e1 - 1]
        dy = y[s2] - y[e1 - 1]

        if np.hypot(dx, dy) > max_dist_px:
            continue

        t = np.linspace(0, 1, gap + 2)
        x[e1 - 1 : s2 + 1] = x[e1 - 1] + dx * t
        y[e1 - 1 : s2 + 1] = y[e1 - 1] + dy * t

    return x, y

# ======================================================
# Pipeline FINAL
# ======================================================
def process_trajectory(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)

    # Nettoyage brut
    x, y = clean_velocity_outliers(x, y)
    x, y = clean_local_spikes(x, y)
    x, y = remove_short_segments(x, y)

    # Début réel du point
    start = find_real_rally_start(x, y)
    x, y = crop_before_rally(x, y, start)

    # Raccords progressifs
    x, y = connect_close_segments(x, y)
    x, y = interpolate_small_gaps_xy(x, y)

    # 🔥 NOUVEAU : trous intermédiaires cohérents
    x, y = connect_medium_gaps(x, y)

    # Micro-finitions
    x, y = connect_very_small_gaps(x, y)

    # Lissage final
    x, y = smooth_segments(x, y)

    return x.tolist(), y.tolist()

# ======================================================
# Extraction principale
# ======================================================
def extract_series(
    ball_data: Dict[str, Any]
) -> Tuple[List[int], List[float], List[float], List[bool], List[str]]:
    frames = sorted(int(k) for k in ball_data.keys())

    xs, ys, visibles, actions = [], [], [], []
    for f in frames:
        d = ball_data[str(f)]
        xs.append(float(d["x"]) if d.get("x") is not None else np.nan)
        ys.append(float(d["y"]) if d.get("y") is not None else np.nan)
        visibles.append(bool(d.get("visible", DEFAULT_VISIBLE)))
        actions.append(str(d.get("action", DEFAULT_ACTION)))

    xs, ys = process_trajectory(xs, ys)

    return frames, xs, ys, visibles, actions



if __name__ == "__main__":
    print("✅ data_loader.py adapté et prêt")


# ======================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Charger un point d'exemple
    point_id = 230
    data_path = Path(r"c:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2")
    json_file = data_path / f"ball_data_{point_id}.json"
    
    if json_file.exists():
        ball_data = load_ball_json(json_file)
        frames, xs, ys, visibles, actions = extract_series(ball_data)
        
        # Trajectoire brute
        xs_raw = [float(ball_data[str(f)]["x"]) if ball_data[str(f)].get("x") is not None else np.nan 
                  for f in frames]
        ys_raw = [float(ball_data[str(f)]["y"]) if ball_data[str(f)].get("y") is not None else np.nan 
                  for f in frames]
        
        # Compter hits et bounces
        hits = [i for i, a in enumerate(actions) if a == "hit"]
        bounces = [i for i, a in enumerate(actions) if a == "bounce"]
        
        print(f"\n📊 Point {point_id}")
        print(f"Points bruts: {sum(~np.isnan(xs_raw))}")
        print(f"Points nettoyés: {sum(~np.isnan(xs))}")
        print(f"Hits: {len(hits)}, Bounces: {len(bounces)}")
        
        # Visualisation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # X trajectory
        ax1.plot(frames, xs_raw, 'o', color='lightgray', markersize=2, label='Brut', alpha=0.5)
        ax1.plot(frames, xs, '-', color='orange', linewidth=2, label='Nettoyé')
        for h in hits:
            ax1.axvline(frames[h], color='green', alpha=0.3, linewidth=1)
        for b in bounces:
            ax1.axvline(frames[b], color='red', alpha=0.3, linewidth=1)
        ax1.set_ylabel('X (pixels)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Trajectoire Point {point_id} - X et Y', fontsize=14, fontweight='bold')
        
        # Y trajectory
        ax2.plot(frames, ys_raw, 'o', color='lightgray', markersize=2, label='Brut', alpha=0.5)
        ax2.plot(frames, ys, '-', color='orange', linewidth=2, label='Nettoyé')
        for h in hits:
            ax2.axvline(frames[h], color='green', alpha=0.3, linewidth=1, label='Hit' if h == hits[0] else '')
        for b in bounces:
            ax2.axvline(frames[b], color='red', alpha=0.3, linewidth=1, label='Bounce' if b == bounces[0] else '')
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Visualisation affichée")
    else:
        print(f"❌ Fichier introuvable: {json_file}")
