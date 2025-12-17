from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from scipy.signal import savgol_filter

# Valeurs par défaut
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
# Nettoyage PUREMENT GÉOMÉTRIQUE (neutre)
# ======================================================
def clean_velocity_outliers(
    x: np.ndarray, y: np.ndarray, max_jump_px: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supprime les téléportations (sauts image impossibles).
    """
    for _ in range(3):
        valid = np.where(~np.isnan(x))[0]
        if len(valid) < 2:
            break

        dx = np.diff(x[valid])
        dy = np.diff(y[valid])
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
    """
    Supprime un point aberrant isolé par rapport à ses voisins.
    """
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < 3:
        return x, y

    to_nan = []
    for i in range(1, len(valid) - 1):
        i0, i1, i2 = valid[i - 1], valid[i], valid[i + 1]
        pred_x = 0.5 * (x[i0] + x[i2])
        pred_y = 0.5 * (y[i0] + y[i2])
        err = np.hypot(x[i1] - pred_x, y[i1] - pred_y)
        if err > threshold_px:
            to_nan.append(i1)

    if to_nan:
        x[to_nan] = np.nan
        y[to_nan] = np.nan

    return x, y


def remove_short_segments(
    x: np.ndarray, y: np.ndarray, min_len: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supprime les segments visibles trop courts (bruit).
    """
    mask = ~np.isnan(x)
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


def interpolate_small_gaps(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """
    Interpole UNIQUEMENT les petits trous (tracking intermittent).
    """
    n = len(arr)
    idx = np.arange(n)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return arr

    interp = np.interp(idx, idx[valid], arr[valid])

    gaps = np.where(np.diff(np.where(valid)[0]) > max_gap)[0]
    for g in gaps:
        i0 = np.where(valid)[0][g] + 1
        i1 = np.where(valid)[0][g + 1]
        interp[i0:i1] = np.nan

    return interp


def smooth_segments(
    x: np.ndarray, y: np.ndarray, window: int = 7, poly: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lissage Savitzky–Golay segment par segment (sans traverser les NaN).
    """
    mask = ~np.isnan(x)
    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        length = e - s
        if length >= window:
            w = window if window % 2 == 1 else window - 1
            try:
                x[s:e] = savgol_filter(x[s:e], w, poly)
                y[s:e] = savgol_filter(y[s:e], w, poly)
            except Exception:
                pass

    return x, y

# ======================================================
# Pipeline neutre complet
# ======================================================
def process_trajectory(
    xs: List[float], ys: List[float]
) -> Tuple[List[float], List[float]]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    x, y = clean_velocity_outliers(x, y)
    x, y = clean_local_spikes(x, y)
    x, y = remove_short_segments(x, y)

    x = interpolate_small_gaps(x)
    y = interpolate_small_gaps(y)

    x, y = smooth_segments(x, y)

    return x.tolist(), y.tolist()

# ======================================================
# Extraction principale
# ======================================================
def extract_series(
    ball_data: Dict[str, Any]
) -> Tuple[List[int], List[float], List[float], List[bool], List[str]]:
    """
    Extraction + nettoyage neutre.
    AUCUNE logique hit / bounce.
    """
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

# ======================================================
if __name__ == "__main__":
    print("data_loader.py prêt")
